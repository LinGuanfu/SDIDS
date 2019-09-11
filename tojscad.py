# -*- coding: UTF-8 -*-
def creatStoreyModeShapeJSCAD(mode,order,dof):
	plate = ''
	plate += "function getParameterDefinitions(){return [{ name: 'numMode', caption: '模态阶数:', type: 'int', initial: 1, min: 1, max: "
	plate += str(order) 
	plate += ", step: 1 }];}function main(params) {var storey = involuteStorey(params.numMode);return storey;}function involuteStorey(numMode) {var storeyMode = ["
	for o in range(order):
		plate += "["
		for d in range(dof):
			plate += "{'dof':" + str(d+1) + ",'value':" + str(mode[o*dof+d]) + "},"
		plate += "],"
	plate += "];var numStorey = "
	plate += str(dof)
	plate += ";var pillarHeight = 10;var storeyWidth = 10;var pillarPitch = 0.5;numMode = numMode - 1;var group = [];for (var k = 0; k < numStorey; k++) {if(k < 1){var pillarLeft = CSG.cylinder({start: [0, -storeyWidth/2, k*1.1*pillarHeight], end: [0, -storeyWidth/2 + storeyWidth*0.4*storeyMode[numMode][k].value*1.05, k*1.1 *pillarHeight + pillarHeight*1.05], radius: pillarPitch, resolution: 200});var pillarRight = CSG.cylinder({start: [0, storeyWidth/2, k*1.1*pillarHeight], end: [0, storeyWidth/2 + storeyWidth*0.4*storeyMode[numMode][k].value*1.05, k*1.1 *pillarHeight + pillarHeight*1.05], radius: pillarPitch, resolution: 200});}else{pillarLeft = CSG.cylinder({start: [0, -storeyWidth/2 + storeyWidth*0.4*storeyMode[numMode][k - 1].value - (storeyWidth*0.4*storeyMode[numMode][k].value - storeyWidth*0.4*storeyMode[numMode][k - 1].value)*0.05, k*1.1*pillarHeight - pillarHeight*0.05], end: [0, -storeyWidth/2 + storeyWidth*0.4*storeyMode[numMode][k].value + (storeyWidth*0.4*storeyMode[numMode][k].value - storeyWidth*0.4*storeyMode[numMode][k - 1].value)*0.05, k*1.1 *pillarHeight + pillarHeight*1.05], radius: pillarPitch, resolution: 200});pillarRight = CSG.cylinder({start: [0, storeyWidth/2 + storeyWidth*0.4*storeyMode[numMode][k - 1].value - (storeyWidth*0.4*storeyMode[numMode][k].value - storeyWidth*0.4*storeyMode[numMode][k - 1].value)*0.05, k*1.1*pillarHeight - pillarHeight*0.05], end: [0, storeyWidth/2 + storeyWidth*0.4*storeyMode[numMode][k].value + (storeyWidth*0.4*storeyMode[numMode][k].value - storeyWidth*0.4*storeyMode[numMode][k - 1].value)*0.05, k*1.1 *pillarHeight + pillarHeight*1.05], radius: pillarPitch, resolution: 200});}var storey = cube({size: [pillarPitch*2.5, storeyWidth*1.2, pillarHeight*0.1],center: [true,true,false]}).translate([0,storeyWidth*0.4*storeyMode[numMode][k].value,pillarHeight+(pillarHeight*1.1)*k]);group.push(pillarLeft,pillarRight,storey)}return group}"

	try:
		with open("./web/models/storeyModeShape.jscad",'w',encoding="utf-8") as f:
			f.write(plate)
		return 1
	except Exception as e:
		return 2


if __name__ == '__main__':
	mode = [1,-1,0.5,0.5,1,-1]
	order = 2
	dof = 3
	creatStoreyModeShapeJSCAD(mode,order,dof)