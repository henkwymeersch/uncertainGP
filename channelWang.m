classdef channelWang < handle
% Markus Froehle, Date: 2019-01-17
% Description: Implementation of ad-hoc channel with desired decorrelation
% properties according to the paper 
% Wang, E. K. Tameh, and A. R. Nix, ?Joint shadowing process in urban peer-to-peer radio channels,? IEEE Trans. Veh. Technol. , vol. 57, no. 1, pp. 52?64, Jan. 2008.
% 
% allows the following function calls:
%   -PRXdB = evaluate(obj, xTX, yTX, xRX, yRX, usePL )
%   -DB = generateNoisyMeasurementDBSigma( obj, NoMeasurements, sigmaTX, sigmaRX, x,usePL )
%   -DB = getMeasurementDB( obj )

    properties
        eta; % Pathloss exponent
        sigmaPsi; % shadowing std.dev. in dB
        dc; % decorrelation distance
        xmax;
        ymax;
        delta_f;
        N;
        c;
        fc;
        lambda;
        delta_x;
        N_table;
        a;
        beta;
        f_T;
        phi;
        f_x;
        f_y;
        f_R;
        f_u;
        f_v;
        f1st;
        M;
        f2nd;
        f;
        theta;
        f_bar;
        delta_theta;
        theta_bar;
        DB;
        L0dB;
    end
    
    methods
        function obj = channelWang(eta, dc, sigmaPsi, L0dB) % constructor
        % function obj = channelWang(eta, dc, sigmaPsi, L0dB) % constructor
        % constructor function with input eta (pathloss exponent), dc
        % (decorrelation distance), sigmaPsi (shadowing std. dev. in dB),
        % L0dB (channel gain (PTX + antenna gain)
        
            obj.eta = eta;
            obj.dc = dc;
            obj.sigmaPsi = sigmaPsi;
            obj.L0dB = L0dB;
            
            % setup
            obj.xmax = 50; % max. dimensions
            obj.ymax = 50;
            obj.delta_f = dc/1000; % frequency spacing, recommended <dc/20
            assert( 1/obj.delta_f > max(obj.xmax, obj.ymax) );
            obj.N=5000; % number of sinoids
            obj.c=3e8; % propagation speed
            obj.fc = 2.4e9; % frequency of transmitted signal
            obj.lambda = obj.c/obj.fc; % wave length
            obj.delta_x = obj.lambda;
            assert( obj.delta_x < 10* obj.lambda);
            obj.N_table = ceil( 1/(obj.delta_x * obj.delta_f) );
            if mod(obj.N_table,2) ~= 0
                obj.N_table = obj.N_table+1;
            end            
            obj.a = log(2)/obj.dc;
            
            
            % channel:
            obj.beta = rand([obj.N_table/2,1]);
            obj.f_T = obj.a/(2*pi) .* sqrt( 1./(1-obj.beta).^2  - 1);
            obj.phi = 2*pi*rand([obj.N_table/2, 1]);
            obj.f_x = obj.f_T .* cos( obj.phi );
            obj.f_y = obj.f_T .* sin( obj.phi );
            
            obj.beta = rand([obj.N_table/2,1]);
            obj.f_R = obj.a/(2*pi) .* sqrt( 1./(1-obj.beta).^2  - 1);
            obj.phi = 2*pi*rand([obj.N_table/2, 1]);
            obj.f_u = obj.f_R .* cos( obj.phi );
            obj.f_v = obj.f_R .* sin( obj.phi );
            
            obj.f1st = [obj.f_x obj.f_y obj.f_u obj.f_v];
            obj.M = [0 0 1 0;0 0 0 1;1 0 0 0;0 1 0 0];
            obj.f2nd = obj.f1st * obj.M;
            
            obj.f = [obj.f1st; obj.f2nd];
            obj.theta = 2*pi*rand([obj.N_table/2, 1]);
            obj.theta = [obj.theta;obj.theta];
            
            obj.f_bar = round( (obj.f + obj.delta_f) / (2*obj.delta_f) ) * (2*obj.delta_f) - obj.delta_f;
            obj.delta_theta = 2 * pi / obj.N_table;
            obj.theta_bar = round( (obj.theta - obj.delta_theta/2 ) / obj.delta_theta ) * obj.delta_theta;

            
        end
        
        function PRXdB = evaluate(obj, xTX, yTX, xRX, yRX, usePL )
        % function PRXdB = evaluate(obj, xTX, yTX, xRX, yRX, usePL )
        % returns the received power in dB for a channel "obj", in location
        % [xTX, yTX, xRX, yRX]; usePL = 1 when path loss is included         
            x = xTX;
            y = yTX;
            u = xRX;
            v = yRX;
            d_TX_RX = sqrt((x-u)^2+(y-v)^2);
            if usePL == 1
                if d_TX_RX == 0 % ignore distance-dependent PL for zero distance
                    PRXdB = obj.L0dB + sum( sqrt(2/obj.N) * obj.sigmaPsi * cos( 2*pi * ( obj.f(:,1)*x ...
                    + obj.f(:,2)*y + obj.f(:,3)*u + obj.f(:,4)*v + obj.theta) ) );
                else
                    PRXdB = obj.L0dB + sum( sqrt(2/obj.N) * obj.sigmaPsi * cos( 2*pi * ( obj.f(:,1)*x ...
                        + obj.f(:,2)*y + obj.f(:,3)*u + obj.f(:,4)*v + obj.theta) ) ) + ...
                        -10*obj.eta*log10( d_TX_RX );
                end
            elseif usePL == 0 % ignore distance-dependent PL for zero distance
                PRXdB = obj.L0dB + sum( sqrt(2/obj.N) * obj.sigmaPsi * cos( 2*pi * ( obj.f(:,1)*x ...
                    + obj.f(:,2)*y + obj.f(:,3)*u + obj.f(:,4)*v + obj.theta) ) );
            elseif usePL == 3 % use only PL no shadowing
                if d_TX_RX == 0
                    PRXdB = obj.L0dB ;
                else
                    PRXdB = obj.L0dB + -10*obj.eta*log10( d_TX_RX );
                end
            end
            assert( ~isinf( PRXdB ) )
        end
        
         function DB = generateNoisyMeasurementDBSigma( obj, NoMeasurements, sigmaTX, sigmaRX, x,usePL )
         % function DB = generateNoisyMeasurementDBSigma( obj, NoMeasurements, sigmaTX, sigmaRX, x,usePL )
         % this function generates NoMeasurements channel measurements with input location uncertainty [sigmaTX, sigmaRX] 
         % and true locations x
            DB.NoMeasurements = NoMeasurements;
            
            y = zeros(NoMeasurements,1);
            for i=1:NoMeasurements
                % evaluate channel at true position x
                y(i) = obj.evaluate( x(i,1), x(i,2), x(i,3), x(i,4), usePL );
            end
            
            sigmaTX = sigmaTX * ones(NoMeasurements/2,1); % make a vector
            sigmaRX = sigmaRX * ones(NoMeasurements/2,1);
            
            % add noise to position estimate consider reciprocity!
            nxTX = randn(NoMeasurements/2,1 ) .* sigmaTX;
            nyTX = randn(NoMeasurements/2,1 ) .* sigmaTX;
            nxRX = randn(NoMeasurements/2,1 ) .* sigmaRX;
            nyRX = randn(NoMeasurements/2,1 ) .* sigmaRX;
            
            muxTX = x(:,1) + [nxTX; nxRX]; % estimate of mean
            muyTX = x(:,2) + [nyTX; nyRX];
            muxRX = x(:,3) + [nxRX; nxTX];
            muyRX = x(:,4) + [nyRX; nyTX];
            
            mu = [muxTX, muyTX, muxRX, muyRX]; % estimated mean position
            
            nxTX = randn(NoMeasurements/2,1 ) .* sigmaTX;
            nyTX = randn(NoMeasurements/2,1 ) .* sigmaTX;
            nxRX = randn(NoMeasurements/2,1 ) .* sigmaRX;
            nyRX = randn(NoMeasurements/2,1 ) .* sigmaRX;
            
            xhatxTX = x(:,1) + [nxTX; nxRX]; % estimate of x
            xhatyTX = x(:,2) + [nyTX; nyRX];
            xhatxRX = x(:,3) + [nxRX; nxTX];
            xhatyRX = x(:,4) + [nyRX; nyTX];
            
            xhat = [xhatxTX, xhatyTX, xhatxRX, xhatyRX]; % estimated position
            
            DB.y = y;       % measurement
            DB.x = x;       % true state (hidden to the algorithm)
            DB.mu = mu;     % mean estimate of true state
            DB.xhat = xhat; % estimate
            DB.sigmaTX = [sigmaTX;sigmaTX] .* ones(DB.NoMeasurements,1);
            DB.sigmaRX = [sigmaRX;sigmaRX] .* ones(DB.NoMeasurements,1);
            obj.DB = DB;
         end               
        
        function DB = getMeasurementDB( obj )
            DB = obj.DB;
        end
    end
    
end

