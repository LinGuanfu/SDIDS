// Here we define the user editable parameters:
function getParameterDefinitions () {
  return [
    { name: 'numStorey', caption: '剪切层数目:', type: 'int', initial: 10, min: 3, max: 20, step: 1 },
    { name: 'pillarHeight', caption: '单层高度:', type: 'float', initial: 10 },
    { name: 'storeyWidth', caption: '剪切层宽度:', type: 'float', initial: 10 },
    // { name: 'clearance', caption: '分辨率:', type: 'float', initial: 0.0, step: 0.1 },
    { name: 'pillarPitch', caption: '柱子直径:', type: 'float', initial: 0.5, step: 0.1 }
  ];
}

function main (params) {
  var storey = involuteStorey(
    params.numStorey,
    params.pillarHeight,
    params.storeyWidth,
    params.clearance,
    params.pillarPitch
  );
  return storey;
}

function involuteStorey(numStorey, pillarHeight, storeyWidth, clearance, pillarPitch) {
  // body...
  if (arguments.length < 3) storeyWidth = 10;
  if (arguments.length < 4) clearance = 0;
  if (arguments.length < 4) pillarPitch = 0.1;

  var group = []
  for (var k = 0; k < numStorey; k++) {
    var pillarLeft = cylinder({r: pillarPitch, 
                               h: pillarHeight, 
                               center: [true,true,false]}).translate([0,-storeyWidth/2,pillarHeight*1.1*k])
    var pillarRight = cylinder({r: pillarPitch, 
                                h: pillarHeight, 
                                center: [true,true,false]}).translate([0,storeyWidth/2,pillarHeight*1.1*k])
    var storey = cube({size: [pillarPitch*2.5, storeyWidth*1.2, pillarHeight*0.1],
                       center: [true,true,false]}).translate([0,0,pillarHeight+(pillarHeight*1.1)*k])
    group.push(pillarLeft,pillarRight,storey)
  }
  return group
}