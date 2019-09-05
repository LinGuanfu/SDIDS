function getParameterDefinitions () {
  return [
    {name: 'tolLen', type: 'float', initial: 100, caption: '梁长度：'},
    {name: 'numElement', type: 'int', initial: 20, caption: '单元数：'},
    {name: 'sectionHeight', type: 'float', initial: 2.5, caption: '截面高度'},
    {name: 'sectionWidth', type: 'float', initial: 5, caption: '截面宽度'},
    {name: 'beamE', type: 'float', initial: 7.1e10, caption: '弹性模量 E'},
    {name: 'beamRho', type: 'float', initial: 2.21e3, caption: '密度 &#961;'},
  ];
}

function main (params) {

var tolLen = params.tolLen;
var numElement = params.numElement;
var sectionHeight = params.sectionHeight;
var sectionWidth = params.sectionWidth;
var typeSupport = params.typeSupport;
var solidBeamEle = Creatbeamele(tolLen,numElement,sectionHeight,sectionWidth,typeSupport);
var solidBeamType= Creatbeamtype(tolLen,numElement,sectionHeight,sectionWidth,typeSupport);
var cantileverBeam=  union(solidBeamType,solidBeamEle);
return cantileverBeam;
}


function Creatbeamele(tolLen, numElement, sectionHeight, sectionWidth, typeSupport) {

  var struc2 = [];
  eLen = tolLen / numElement;
  for (var j = 1; j < numElement; j++) {
    var result = CSG.cube({ radius: [eLen-eLen/50, sectionWidth, sectionHeight] }).translate([-tolLen / 2 + j * eLen - eLen/50, 0, sectionHeight * 10]);
    var result2 = CSG.cube({ radius: [eLen/50, sectionWidth*1.01, sectionHeight*1.01] }).translate([-tolLen / 2 +(eLen-eLen/50)+ (j-1) * eLen , 0, sectionHeight * 10]).setColor([55,55,55]);
    struc2.push( result2,result);}
    return struc2;
  
}

function Creatbeamtype(tolLen,numElement,sectionHeight,sectionWidth,typeSupport){
  eLen=tolLen/numElement;
 
  var support=CSG.cube({radius: [eLen/2,sectionWidth*4, sectionHeight*10]}).translate([-tolLen/2-eLen/2, 0, sectionHeight*10]);
  return support.setColor([0,0,0]);
}



///  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%