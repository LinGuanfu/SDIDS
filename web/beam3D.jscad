

function getParameterDefinitions () {
  return [
    {name: 'Len', type: 'float', initial: 100, caption: '长度：'},
    {name: 'ele', type: 'int', initial: 20, caption: '单元数：'},
    {name: 'height', type: 'float', initial: 2.5, caption: '截面高度'},
    {name: 'width', type: 'float', initial: 5, caption: '截面宽度'},
  ];
}

function main (params) {

var Len = params.Len;
var ele = params.ele;
var height = params.height;
var width = params.width;
var type_support = params.type_support;
var solidbeamele = Creatbeamele(Len,ele,height,width,type_support);
var solidbeamtype= Creatbeamtype(Len,ele,height,width,type_support);
var struc=  union(solidbeamtype,solidbeamele);
return struc;
}


function Creatbeamele(Len, ele, height, width, type_support) {

  var struc2 = [];
  eLen = Len / ele;
  for (var j = 1; j < ele; j++) {
    var result = CSG.cube({ radius: [eLen-eLen/50, width, height] }).translate([-Len / 2 + j * eLen - eLen/50, 0, height * 10]);
    var result2 = CSG.cube({ radius: [eLen/50, width*1.01, height*1.01] }).translate([-Len / 2 +(eLen-eLen/50)+ (j-1) * eLen , 0, height * 10]).setColor([55,55,55]);
    struc2.push( result2,result);}
    return struc2;
  
}

function Creatbeamtype(Len,ele,height,width,type_support){
  eLen=Len/ele;
 
  var support=CSG.cube({radius: [eLen/2,width*4, height*10]     }).translate([-Len/2-eLen/2, 0, height*10]);
  return support.setColor([0,0,0]);
}



///  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%