function x = expansion(hdl, x),
% Alternative expansion function, used by the demo script
% sfa_tk/demo/expansion_demo.m
  
  fprintf('******* sfa-tk is using a user-defined EXPANSION function!\n');
  x = cat(2, x, x.^4);
