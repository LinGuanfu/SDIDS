function goBarChartH(dataArr){
  // 声明所需变量
  var canvas,ctx;
  // 图表属性
  var cWidth, cHeight, cMargin, cSpace;
  var originX, originY;
  // 柱状图属性
  var bMargin, tobalBars, bWidth, maxValue;
  var totalYNomber;
  var gradient;

  // 运动相关变量
  var ctr, numctr, speed;
  //鼠标移动
  var mousePosition = {};

  // 获得canvas上下文
  canvas = document.getElementById("barChart");


  if(canvas && canvas.getContext){
    ctx = canvas.getContext("2d");
  }
  initChart(); // 图表初始化
  drawLineLabelMarkers(); // 绘制图表轴、标签和标记
  drawBarAnimate(); // 绘制柱状图的动画

  //检测鼠标移动
  // var mouseTimer = null;
  // canvas.addEventListener("mousemove",function(e){
  //     e = e || window.event;
  //     if( e.offsetX || e.offsetX==0 ){
  //         mousePosition.x = e.offsetX;
  //         mousePosition.y = e.offsetY;
  //     }else if( e.layerX || e.layerX==0 ){
  //         mousePosition.x = e.layerX;
  //         mousePosition.y = e.layerY;
  //     }
      
  //     clearTimeout(mouseTimer);
  //     mouseTimer = setTimeout(function(){
  //         ctx.clearRect(0,0,canvas.width, canvas.height);
  //         drawLineLabelMarkers();
  //         drawBarAnimate(true);
  //     },10);
  // });

  //点击刷新图表
  // canvas.onclick = function(){
  //   initChart(); // 图表初始化
  //   drawLineLabelMarkers(); // 绘制图表轴、标签和标记
  //   drawBarAnimate(); // 绘制折线图的动画
  // };


  // 图表初始化
  function initChart(){
    // 图表信息
    cMargin = 0;
    cSpace = 80;
    /*这里是对高清屏幕的处理，
         方法：先将canvas的width 和height设置成本来的两倍
         然后将style.height 和 style.width设置成本来的宽高
         这样相当于把两倍的东西缩放到原来的 1/2，这样在高清屏幕上 一个像素的位置就可以有两个像素的值
         这样需要注意的是所有的宽高间距，文字大小等都得设置成原来的两倍才可以。
    */

    var dheight = window.getComputedStyle(drawbard).height; // 获取div  id="boxdrawing" 的长宽
    var dwidth = window.getComputedStyle(drawbard).width;
    var width = dwidth.split("px")[0] * 9 / 10;
    var height = dheight.split("px")[0];
    canvas.width = width * 2 ;
    // canvas.height = 480* 2;
    canvas.height = canvas.width*0.618;
    canvas.style.height = canvas.height/2 + "px";
    canvas.style.width = canvas.width/2 + "px";
    cHeight = canvas.height - cMargin - cSpace;
    cWidth = canvas.width - cMargin - cSpace;
    originX = cMargin + cSpace;
    originY = cMargin + cHeight;

    // 柱状图信息
    bMargin = canvas.width/40;
    tobalBars = dataArr.length;
    bWidth = parseInt( cWidth/tobalBars - bMargin );
    maxValue = 0;
    for(var i=0; i<dataArr.length; i++){
      console.log('hello');
      // var barVal = parseInt( dataArr[i][1] );
      var barVal = dataArr[i][1];
      console.log(barVal);
      if( barVal > maxValue ){
          maxValue = barVal;
      }
    }
    console.log(maxValue)
    maxValue += 0.1*maxValue;
    totalYNomber = 1;
    // 运动相关
    ctr = 1;
    numctr = 10;
    speed = 10;

    //柱状图渐变色
    gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'green');
    gradient.addColorStop(1, 'rgba(67,203,36,1)');

  }

  // 绘制图表轴、标签和标记
  function drawLineLabelMarkers(){
    ctx.translate(0.5,0.5);  // 当只绘制1像素的线的时候，坐标点需要偏移，这样才能画出1像素实线
    ctx.font = "24px Arial";
    ctx.lineWidth = 2;
    ctx.fillStyle = "#000";
    ctx.strokeStyle = "#000";
    // y轴
    drawLine(originX, originY, originX, cMargin);
    // x轴
    drawLine(originX, originY, originX+cWidth, originY);

    // 绘制标记
    drawMarkers();
    ctx.translate(-0.5,-0.5);  // 还原位置
  }

  // 画线的方法
  function drawLine(x, y, X, Y){
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(X, Y);
    ctx.stroke();
    ctx.closePath();
  }

  // 绘制标记
  function drawMarkers(){
    ctx.strokeStyle = "#000";
    // 绘制 y
    var oneVal = parseInt(maxValue/totalYNomber);
    ctx.textAlign = "right";
    for(var i=0; i<=totalYNomber; i++){
        var markerVal =  i*oneVal;
        var xMarker = originX-10;
        var yMarker = parseInt( cHeight*(1-markerVal/maxValue) ) + cMargin;
        
        ctx.fillText(markerVal, xMarker, yMarker+3, cSpace); // 文字
        if(i>0){
            drawLine(originX+2, yMarker, originX+cWidth, yMarker);
        }
    }
    // 绘制 x
    ctx.textAlign = "center";
    for(var i=0; i<tobalBars; i++){
        var markerVal = dataArr[i][0];
        var xMarker = parseInt( originX+cWidth*(i/tobalBars)+bMargin+bWidth/2 );
        var yMarker = originY+30;
        ctx.fillText(markerVal, xMarker, yMarker, cSpace); // 文字
    }
    // 绘制标题 y
    ctx.save();
    ctx.rotate(-Math.PI/2);
    ctx.fillText("损伤结果", -canvas.height/2-bWidth/5, cSpace-bWidth/5);
    ctx.restore();
    // 绘制标题 x
    ctx.fillText("单元", originX+cWidth/2, originY+cSpace/2+30);
  };

  //绘制柱形图
  function drawBarAnimate(mouseMove){
    for(var i=0; i<tobalBars; i++){
      var oneVal = parseInt(maxValue/totalYNomber);
      var barVal = Math.floor(dataArr[i][1]*100)/100;                               ///////////显示的结果图  上标的数据
      var barH = parseInt( cHeight*barVal/maxValue * ctr/numctr );
      var y = originY - barH;
      var x = originX + (bWidth+bMargin)*i + bMargin;
      var xMarker = parseInt( originX+cWidth*(i/tobalBars)+bMargin+bWidth/2 );
      drawRect( x, y, bWidth, barH-1, mouseMove );  //高度减一避免盖住x轴
      ctx.fillStyle = 'black';
      ctx.fillText((barVal*ctr/numctr), xMarker, y-8); // 文字
    }
    if(ctr<numctr){
      ctr++;
      setTimeout(function(){
          ctx.clearRect(0,0,canvas.width, canvas.height);
          drawLineLabelMarkers();
          drawBarAnimate();
      }, speed);
    }
  }

  //绘制方块
  function drawRect( x, y, X, Y, mouseMove ){
    ctx.beginPath();
    ctx.rect( x, y, X, Y );
    ctx.fillStyle = gradient;
    ctx.strokeStyle = gradient;

         //if(mouseMove && ctx.isPointInPath(mousePosition.x*2, mousePosition.y*2)){ //如果是鼠标移动的到柱状图上，重新绘制图表

    //     if( ctx.isPointInPath(mousePosition.x*2, mousePosition.y*2)){ //如果是鼠标移动的到柱状图上，重新绘制图表
    //     ctx.fillStyle = "green";
        
    // }else{
    //     ctx.fillStyle = gradient;
    //     ctx.strokeStyle = gradient;
    // }

    ctx.fill();
    ctx.closePath();

  }


}
