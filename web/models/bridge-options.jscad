// Here we define the user editable parameters:
function getParameterDefinitions() {
  return [
    { name: 'numBay', caption: '桥跨数目:', type: 'int', initial: 3, min: 1, max: 10, step: 1 },
    { name: 'pillarHeight', caption: '柱子高度:', type: 'float', initial: 8 },
    { name: 'bridgeWidth', caption: '桥梁宽度:', type: 'float', initial: 5 },
    // { name: 'clearance', caption: '分辨率:', type: 'float', initial: 0.0, step: 0.1 },
    { name: 'spanLength', caption: '单跨长度:', type: 'float', initial: 20, step: 1 }
  ];
}

function main(params) {
  var bridge = involuteBridge(
    params.numBay + 1,
    params.pillarHeight,
    params.bridgeWidth,
    params.clearance,
    params.spanLength
  );
  return bridge;
}

function involuteBridge(numBay, pillarHeight, bridgeWidth, clearance, spanLength) {
  // body...
  if (arguments.length < 3) bridgeWidth = 10;
  if (arguments.length < 4) clearance = 0;
  if (arguments.length < 4) spanLength = 0.1;

  var group = []
  for (var k = 0; k < numBay; k++) {
    var pillarLeft = cylinder({r: 0.075*bridgeWidth, 
                               h: pillarHeight, 
                               center: [true,true,false]}).translate([-spanLength*(numBay-1)/2+spanLength*k,-bridgeWidth*0.35,0]);
    var pillarRight = cylinder({r: 0.075*bridgeWidth, 
                                h: pillarHeight, 
                                center: [true,true,false]}).translate([-spanLength*(numBay-1)/2+spanLength*k,bridgeWidth*0.35,0]);
    var pier = union(
    	cube({size: [0.15*bridgeWidth, bridgeWidth*1.1, pillarHeight*0.1],
                   center: [true,true,false]}).translate([-spanLength*(numBay-1)/2+spanLength*k,0,pillarHeight]),
    	cube({size: [0.15*bridgeWidth, bridgeWidth*0.05, pillarHeight*0.15],
                   center: [true,true,false]}).translate([-spanLength*(numBay-1)/2+spanLength*k,-0.525*bridgeWidth,pillarHeight]),
  		cube({size: [0.15*bridgeWidth, bridgeWidth*0.05, pillarHeight*0.15],
                   center: [true,true,false]}).translate([-spanLength*(numBay-1)/2+spanLength*k, 0.525*bridgeWidth,pillarHeight])
    	);
    if (k == numBay-1) {
      group.push(pillarLeft,pillarRight,pier);
    } else　{
      var bridge = cube({size: [spanLength*0.999, bridgeWidth, pillarHeight*0.1],
                   center: [false,true,false]}).translate([-spanLength*(numBay-1)/2+spanLength*k,0,pillarHeight*1.1]);
      group.push(pillarLeft,pillarRight,pier,bridge);
    }
  }
  return group
}