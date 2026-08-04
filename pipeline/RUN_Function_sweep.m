function [CL, CD, q_stag, Kn] = RUN_Function_sweep(stl_path, altitude_km, velocity_ms, alpha_deg, model, Twall)
% Callable wrapper around FOSTRAD for batch sweeps.
% Args:
%   stl_path     - absolute path to binary .stl file (string)
%   altitude_km  - altitude in km  (scalar)
%   velocity_ms  - freestream velocity in m/s (scalar)
%   alpha_deg    - angle of attack in degrees (scalar)
%   model        - heat flux model string: 'sc', 'krd', 'fr', or 'vd'
%   Twall        - wall temperature in K (scalar)

cDir = fileparts(mfilename('fullpath'));

addpath(genpath(fullfile(cDir, 'STRATH_A_main')));
addpath(genpath(fullfile(cDir, 'DataSets_SM')));
addpath(genpath(fullfile(cDir, 'External_Functions')));
addpath(genpath(fullfile(cDir, 'Internal_Functions')));
addpath(genpath(fullfile(cDir, 'Mesh_voxelisation')));
addpath(genpath(fullfile(cDir, 'RadiusSmoothFunc')));

STLnames = {stl_path};

opt.LREF          = [];
opt.SREF          = [];
opt.scale         = 1.0;
opt.autoSREFFlag  = 0;
opt.checks        = 0;      % MUST be 0 — disables plots for batch runs
opt.fixed_COG     = [];
opt.NminFaces     = 100;
opt.NmaxFaces     = 500;
opt.nPix          = [1500 1501];   % lower res than default — faster for sweeps
opt.nPixStep      = 250;
opt.bfcFlag       = 1;
opt.NnormGrid     = 150;
opt.voxFlag       = 0;
opt.AeroFlag      = 1;
opt.ThermoFlag    = 1;
opt.ablationFlag  = 0;
opt.RN_const_flag = 0;
opt.rN_ref        = [];
opt.radiusUpdateFlag = 1;
opt.Twall         = Twall;
opt.AccCoeff      = 1;
opt.NsmoothRadius = 3;
opt.FlatEdge      = 0.9;
opt.RmaxRef       = [];
opt.AeroThModel   = model;
opt.CF_Flag       = 0;
opt.cordLengthRef = [];
opt.MESH          = [];

BodyPar.SimpleGeom              = [1];
BodyPar.rho                     = [2700];
BodyPar.Q_ampl                  = [1];
BodyPar.ShellFlag               = [0];
BodyPar.ThinObjFlag             = [0];
BodyPar.shellThickness          = [0.1];
BodyPar.survingFaceID_sys_global = cell(1,1);
BodyPar.ablFaceID_sys_global     = cell(1,1);

Roll = 0;
SS   = 0;

[strath, ~, opt] = STRATH_A_mb(STLnames, altitude_km, velocity_ms, Roll, alpha_deg, SS, BodyPar, opt);

CL = strath{4};
CD = strath{3};
Kn = strath{2};

if numel(CL) > 1, CL = CL(1); end
if numel(CD) > 1, CD = CD(1); end
if numel(Kn) > 1, Kn = Kn(1); end

StConst = strath{9};
Stmap   = strath{11};
if iscell(Stmap)
    q_stag = max(Stmap{1}) * StConst(1);
else
    q_stag = max(Stmap) * StConst(1);
end

end
