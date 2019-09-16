

function CreatModel() {
  $("svg").empty();  // 重置画面

  // 悬臂梁模型数据    数据接口需要考虑
  var model1 = [0, 0.000174994128944424, 0.000696749976453364, 0.00156042776051443, 0.00276118779229501, 0.00429419058281364, 0.00615459699128239, 0.00833756841452226, 0.0108382670168536, 0.0136518559998661, 0.0167734999114701, 0.0201983649936167, 0.0239216195681028, 0.0279384344598673, 0.0322439834571755, 0.0368334438080932, 0.0417019967526534, 0.0468448280901240, 0.0522571287807759, 0.0579340955815583, 0.0638709317150906, 0.0700628475713706, 0.0765050614416096, 0.0831928002836050, 0.0901213005180483, 0.0972858088551885, 0.104681583151257, 0.112303893294078, 0.120148022117260, 0.128209266342401, 0.136482937548706, 0.144964363169447, 0.153648887514690, 0.162531872819700, 0.171608700318456, 0.180874771341712, 0.190325508439016, 0.199956356524142, 0.209762784043319, 0.219740284165746, 0.229884375995820, 0.240190605806433, 0.250654548292959, 0.261271807847259, 0.272038019851173, 0.282948851988972, 0.294000005578234, 0.305187216918582, 0.316506258657789, 0.327952941174707, 0.339523113978462, 0.351212667123454, 0.363017532639657, 0.374933685977627, 0.386957147467733, 0.399083983793215, 0.411310309476509, 0.423632288378295, 0.436046135208908, 0.448548117051545, 0.461134554896838, 0.473801825188311, 0.486546361378302, 0.499364655493861, 0.512253259712209, 0.525208787945270, 0.538227917432891, 0.551307390344452, 0.564444015388343, 0.577634669428761, 0.590876299109632, 0.604165922485240, 0.617500630657136, 0.630877589416991, 0.644294040895028, 0.657747305213646, 0.671234782145915, 0.684753952778743, 0.698302381180238, 0.711877716070887, 0.725477692498390, 0.739100133515854, 0.752742951863026, 0.766404151650314, 0.780081830045373, 0.793774178961878, 0.807479486750331, 0.821196139890643, 0.834922624686461, 0.848657528960919, 0.862399543753493, 0.876147465017941, 0.889900195321156, 0.903656745542738, 0.917416236575109, 0.931177901024053, 0.944941084909720, 0.958705249367874, 0.972469972351328, 0.986234950331483, 1];
  var model2 = [0, -0.00108416758786665, -0.00426644321613536, -0.00944149270376381, -0.0165040041966410, -0.0253487132082044, -0.0358704368240851, -0.0479641165597203, -0.0615248693602901, -0.0764480462329429, -0.0926292980021246, -0.109964647679958, -0.128350568945127, -0.147684070225491, -0.167862783881839, -0.188785059992853, -0.210350064244303, -0.232457879429031, -0.255009610068257, -0.277907489669150, -0.301054990138612, -0.324356932878698, -0.347719601095055, -0.371050852856425, -0.394260234450255, -0.417259093587087, -0.439960692014715, -0.462280317111800, -0.484135392039841, -0.505445584042369, -0.526132910490413, -0.546121842284371, -0.565339404233760, -0.583715272048199, -0.601181865585364, -0.617674438014584, -0.633131160568083, -0.647493202565477, -0.660704806411498, -0.672713357281286, -0.683469447222582, -0.692926933419599, -0.701042990378468, -0.707778155810463, -0.713096370005067, -0.716965008501279, -0.719354907882176, -0.720240384534246, -0.719599246229982, -0.717412796409110, -0.713665831050622, -0.708346628044894, -0.701446928991974, -0.692961913369095, -0.682890165027438, -0.671233630994692, -0.657997572576467, -0.643190508765842, -0.626824151986632, -0.608913336211819, -0.589475937513796, -0.568532787118487, -0.546107577050015, -0.522226758466865, -0.496919432804644, -0.470217235853835, -0.442154214913585, -0.412766699175012, -0.382093163499346, -0.350174085767146, -0.317051797985240, -0.282770331347919, -0.247375255457671, -0.210913511918990, -0.173433242526546, -0.134983612275642, -0.0956146274283069, -0.0553769488734633, -0.0143217010240428, 0.0274997235031015, 0.0700358631765596, 0.113235390559586, 0.157047318789527, 0.201421211278426, 0.246307394387380, 0.291657172829926, 0.337423047563457, 0.383558935932451, 0.430020393833153, 0.476764839676063, 0.523751779930616, 0.570943036045482, 0.618302972548140, 0.665798726138511, 0.713400435603713, 0.761081472394664, 0.808818671719803, 0.856592564026752, 0.904387606759621, 0.952192416297658, 1];
  var model3 = [0, 0.00300415423847113, 0.0116938088501361, 0.0255848925379716, 0.0441938121343178, 0.0670379783033577, 0.0936365108365372, 0.123511105175440, 0.156187041862688, 0.191194320720405, 0.228068901693011, 0.266354034471437, 0.305601659243297, 0.345373861191489, 0.385244361694890, 0.424800029571126, 0.463642396144554, 0.501389158423014, 0.537675655224949, 0.572156301713756, 0.604505968467590, 0.634421291938348, 0.661621903932014, 0.685851568570526, 0.706879216069939, 0.724499863587838, 0.738535414349973, 0.748835327258440, 0.755277150206347, 0.757766911371810, 0.756239363832716, 0.750658079926699, 0.741015392873215, 0.727332184270779, 0.709657517176149, 0.688068115557960, 0.662667691988975, 0.633586126493613, 0.600978500493158, 0.565023990786592, 0.525924629463171, 0.483903936558693, 0.439205433138660, 0.392091043306155, 0.342839394394392, 0.291744025303443, 0.239111513575820, 0.185259532372684, 0.130514849008559, 0.0752112771239775, 0.0196875949215519, -0.0357145578407179, -0.0906527912163834, -0.144786071806676, -0.197776811846312, -0.249292927670032, -0.299009854722740, -0.346612506485061, -0.391797164971885, -0.434273290822532, -0.473765241432334, -0.510013886077244, -0.542778107549190, -0.571836180447441, -0.596987016957011, -0.618051271682295, -0.634872297887712, -0.647316948323866, -0.655276214682743, -0.658665700617759, -0.657425924181406, -0.651522446470199, -0.640945824213533, -0.625711384995478, -0.605858824747897, -0.581451628096438, -0.552576313066475, -0.519341502560482, -0.481876825894889, -0.440331654524400, -0.394873676879594, -0.345687317994592, -0.292972010297984, -0.236940322576879, -0.177815954692801, -0.115831606128949, -0.0512267268703567, 0.0157548405398311, 0.0848673186677393, 0.155865543313523, 0.228507592754114, 0.302557488788948, 0.377787960396337, 0.453983261035133, 0.530942030958151, 0.608480196343531, 0.686433897598104, 0.764662439843953, 0.843051259369453, 0.921514900707463, 1];
  var model4 = [0, -0.00582352712457158, -0.0224079666237983, -0.0484251477630154, -0.0825504050422177, -0.123466348926068, -0.169867821930616, -0.220467847621049, -0.274004381887010, -0.329247678538997, -0.385008084929149, -0.440144088030744, -0.493570437277897, -0.544266177505116, -0.591282433548600, -0.633749797462992, -0.670885179834033, -0.701997998270279, -0.726495588758240, -0.743887739068576, -0.753790257692423, -0.755927506739492, -0.750133842700715, -0.736353924820432, -0.714641866872860, -0.685159224232172, -0.648171824098107, -0.604045462414503, -0.553240506232697, -0.496305454857693, -0.433869526912798, -0.366634353318257, -0.295364867956211, -0.220879498358952, -0.144039767994165, -0.0657394295256648, 0.0131067452834637, 0.0915743886570987, 0.168741039445411, 0.243697590479905, 0.315559546084225, 0.383477957125851, 0.446649904591288, 0.504328407842873, 0.555831640434914, 0.600551344525639, 0.637960344425514, 0.667619070554017, 0.689181016905939, 0.702397067906048, 0.707118644107632, 0.703299630398478, 0.690997065043598, 0.670370582839685, 0.641680620703971, 0.605285408990396, 0.561636786527472, 0.511274891639299, 0.454821795068288, 0.392974153595274, 0.326494975109742, 0.256204596761221, 0.182970987503055, 0.107699494704320, 0.0313221614516695, -0.0452132533833753, -0.120954416337329, -0.194955583540110, -0.266288648964471, -0.334053938767098, -0.397390616317429, -0.455486566661244, -0.507587634829263, -0.553006099507689, -0.591128272037406, -0.621421120403386, -0.643437828669967, -0.656822214081623, -0.661311936626903, -0.656740449080302, -0.643037649220246, -0.620229209894077, -0.588434576667183, -0.547863636762589, -0.498812076665959, -0.441655458965477, -0.376842061502607, -0.304884533555241, -0.226350434378125, -0.141851728804406, -0.0520333226136922, 0.0424392731627229, 0.140893055029503, 0.242661319946749, 0.347098040341559, 0.453592724110298, 0.561585665544086, 0.670583498309621, 0.780174969050238, 0.890046859872835, 1];
  var model5 = [0, 0.00952208007732500, 0.0362053286502983, 0.0772298849872167, 0.129791304888739, 0.191116760758191, 0.258485852039379, 0.329254919834459, 0.400883780688933, 0.470963825345570, 0.537246469691432, 0.597670997702642, 0.650390900069529, 0.693797887127314, 0.726542840124755, 0.747553059813422, 0.756045274616474, 0.751533980766113, 0.733834802096300, 0.703062675785484, 0.659624790282487, 0.604208320854264, 0.537763124587124, 0.461479668172690, 0.376762566417941, 0.285200205236331, 0.188531008153768, 0.0886069785457419, -0.0126447904294555, -0.113261901002825, -0.211286963965298, -0.304807658665144, -0.391995760643739, -0.471144375653343, -0.540702652273150, -0.599307293636164, -0.645810250897013, -0.679302055728744, -0.699130334829113, -0.704913144392065, -0.696546864849645, -0.674208503852176, -0.638352366238526, -0.589701161488774, -0.529231729530181, -0.458155672623928, -0.377895282230435, -0.290055243211285, -0.196390681599253, -0.0987721947840303, 0.000851437149259802, 0.100492114225912, 0.198162077703017, 0.291913578021564, 0.379877758240630, 0.460301977542560, 0.531584830798836, 0.592308166457176, 0.641265465219893, 0.677486014934275, 0.700254401380014, 0.709124928508640, 0.703930683332690, 0.684787068042050, 0.652089732907404, 0.606506955927494, 0.548966626755141, 0.480638100977125, 0.402909294164427, 0.317359481250670, 0.225728353821451, 0.129881964078796, 0.0317762481075394, -0.0665811286140220, -0.163169824716312, -0.255995976124198, -0.343130644525940, -0.422746767254519, -0.493153840559183, -0.552829608777978, -0.600448086265162, -0.634903305891489, -0.655328266049255, -0.661108635661538, -0.651890871789589, -0.627584504996983, -0.588358451402213, -0.534631315036482, -0.467055747329440, -0.386497029892122, -0.294006139867454, -0.190787641675306, -0.0781628228157529, 0.0424714475709205, 0.169687612350150, 0.302075517275762, 0.438293132096087, 0.577119341984131, 0.717508276750712, 0.858644693690640, 1];
  var model6 = [0, -0.0140680343352209, -0.0528351288432641, -0.111160925540233, -0.183955066812219, -0.266228428067186, -0.353157261827608, -0.440155815721979, -0.522953135501772, -0.597669976969629, -0.660892031316683, -0.709736016973380, -0.741905604436678, -0.755734612359761, -0.750215434481936, -0.725011216466772, -0.680450886323913, -0.617506737420162, -0.537754853940958, -0.443319239579590, -0.336801046000831, -0.221194783802756, -0.0997938220725785, 0.0239121683358586, 0.146344907483982, 0.263947240461762, 0.373291739723097, 0.471185021096003, 0.554764733917602, 0.621586393270716, 0.669697515856447, 0.697696891619212, 0.704777259574239, 0.690750145207614, 0.656052143583050, 0.601732481093966, 0.529422243171404, 0.441286198609489, 0.339958667213724, 0.228465350830030, 0.110133464248041, -0.0115071504160887, -0.132828974798006, -0.250214700106234, -0.360164830305302, -0.459401794343967, -0.544967463615594, -0.614311167662524, -0.665365584459390, -0.696608243009088, -0.707106804672143, -0.696546772896806, -0.665240804418236, -0.614119342988434, -0.544702852939565, -0.459056477798359, -0.359728472489597, -0.249674240753558, -0.132168237887055, -0.0107063600295850, 0.111098275811632, 0.229623060402578, 0.341343925032054, 0.442940465399704, 0.531395031061966, 0.604082835500135, 0.658850410417700, 0.694080076324194, 0.708738519459471, 0.702408040340866, 0.675299557555977, 0.628246996408126, 0.562683249576313, 0.480598449543282, 0.384481823816357, 0.277248898190265, 0.162156256032591, 0.0427064397029887, -0.0774541172754591, -0.194641020458100, -0.305238472595258, -0.405803191360851, -0.493162479182089, -0.564503390572309, -0.617450182198493, -0.650127549876204, -0.661207544259172, -0.649938500632158, -0.616154803420594, -0.560266816882415, -0.483230832879310, -0.386499397024315, -0.271952858216938, -0.141813426133051, 0.00145660003944568, 0.155270465962812, 0.317044282027306, 0.484333741933173, 0.654977957129834, 0.827248249742333, 1];
  model = [model1, model2, model3, model4, model5, model6]


  //  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  var modelNmu = parseInt(document.getElementById("NumCreatModel").value); //获取阶数 
  var modelData = model[modelNmu];
  var node = modelData.length;  // 模态包括了约束节点 如这里100个单元，有101个节点  每个节点对应一个坐标
  var numElement = node - 1;
  var ctx = document.getElementById("model");  // 获取svg的元素
  var dwidth = window.getComputedStyle(drawModel).width;
  var width = dwidth.split("px")[0];  //变成数值
  var height = width * 0.30;
  ctx.setAttribute("width", width);  // 实际大小是 width, height  而viewBox则是在这个实际大小里面划分成多少个小格子，用于排版
  ctx.setAttribute("height", height);
  ctx.setAttribute("viewBox", [0, 0, width, height]);


  var mart = [];
  for (var j = 0; j < node; j++) {
    mart[j] = {
      'zx': 8 / 10 * width / numElement * (j) + 1 / 10 * width,
      'zy': -modelData[j] * (2 / 5) * height + 1 / 2 * height
    };  // JS数组从 零 开始
  }

  var mart2 = mart.slice(1, node);  // 错位 从第二节点坐标开始 进行画图
  for (j = 0; j < node - 1; j++) {
    var line = document.createElementNS("http://www.w3.org/2000/svg", "line");//两点一条线
    line.setAttribute("x1", mart[j].zx);
    line.setAttribute("y1", mart[j].zy);
    line.setAttribute("x2", mart2[j].zx);
    line.setAttribute("y2", mart2[j].zy);
    line.setAttribute("stroke", "rgb(0, 125, 255)");
    line.setAttribute("stroke-width", "5");
    // line.setAttribute("style", "fill:gradient");
    ctx.appendChild(line);
  }

  //   var path = document.createElementNS("http://www.w3.org/2000/svg", "path");// path
  //   var pathModel = " M " + mart[0].zx + ' ' + mart[0].zy;
  //   for (let i = 1; i < node - 1; i++) {
  //     pathModel = pathModel + " L " + mart[i].zx + ' ' + mart[i].zy
  //   } // 得到的是一个累加的多重定向path
  //   //  console.log(pathModel); 
  //   path.setAttribute("stroke-width", "5");
  //   path.setAttribute("d", pathModel);
  //   path.setAttribute("stroke", "rgb(0, 255, 255)");
  //  // path.setAttribute("style", "fill:gray");
  //   ctx.appendChild(path);

} // 画图函数结束

$(window).resize(resizesvg);  // 自适应窗口 实际上就是 重新画图
function resizesvg() {
  $("svg").empty()
  CreatModel();
};
CreatModel();  //初始页面状态的画图  即时没有 onclick下，也会画图

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function table() {
  if (document.getElementById("Num").value == "" || document.getElementById("Num").value.search("^[0-9]*$") == -1) {
      document.getElementById("errmsg1").style.display = "block";//判断payNum是否为空或不是数字    提示错误
      document.getElementById("errmsg1").innerHTML = "提示信息：行数为空或不是数字！";
  }
  else {
      document.getElementById("errmsg1").style.display = "none";//隐藏提示信息
      var Num = parseInt(document.getElementById("Num").value); //获取分期数           
      var flag = true;
      var data = "";
      data += " <table >";
      data += " <tr>" +
                  "<td >we are</td>" +
                  "<td >zhuzhu</td>" +
                  "<td >dudu</td>" +                            
              "</tr>" ;
      for (var i = 1; i <= Num; i++) {                   
          data += "<tr >";
          data += "<td>" + i + "</td>";
          data += "<td><input name='ColdDay"+i+"' type='text' class='input'></td>";
          data += "<td><input name='ColdCureMethod"+i+"' type='text' class='input'></td>";                                  
          data += "</tr>";
      }
      
      data += "</table>";
      document.getElementById("div1").style.display = "block";
      document.getElementById("table1").innerHTML = data;
  }
} 







function Creattable() {

  // 悬臂梁模型数据    数据接口需要考虑
  var FreDamage = [10, 20, 30, 40, 50, 60];
  var FreUnDamage = [8, 19, 27, 38, 47, 57];
  var FreChange=[]

  for (var j = 1; j <= FreUnDamage.length; j++) {
    FreChange[j]=Math.floor( (FreDamage[j]-FreUnDamage[j])/FreUnDamage[j]*10000)/100;  
  }   // 表格内容 

  //%%%%%%%%%%%%%%%%%%

  var NumCreattable = parseInt(document.getElementById("NumCreattable").value); //获取阶数数据          
  var data = "";
  data += " <thead class='tableColor'> " +
    " <tr> " +
    "<th>阶次</th>" +
    "<th>W_u-原始固有频率-Hz</th>" +
    "<th>W_d-损伤后固有频率-Hz</th>" +
    "<th>" + "(W_d-W_u)/W_u(%)" + "</th>" +
    " </tr>" +
    "</thead>";   // 创造表头 


  data += " <tbody class='tableColor'> ";  //表格内容定义

  for (var i = 1; i <= NumCreattable; i++) {
    data += "<tr >";
    data += "<td>" + i + "</td>";
    data += "<td>" + FreUnDamage[i] + "</td>";
    data += "<td>" + FreDamage[i] + "</td>";
    data += "<td>" + FreChange[i] + "</td>";
    data += "</tr>";
  }   // 表格内容 

  data += "</tbody>";  // 表格内容定义结束 
  document.getElementById("seleData").innerHTML = data;
} 
Creattable() // 初始状态也要画图










      {/* <table class="table text-center text-light " id="seleData" style="margin-top: 0">
        <thead class="tableColor">
          <tr>
            <th>阶次</th>
            <th>原始固有频率-Hz</th>
            <th>损伤后固有频率-Hz</th>
            <th>频率变化值-Hz</th>
          </tr>
        </thead>
        <tbody class="tableColor">
          <tr>
            <td>1</td>
            <td>10</td>
            <td>10.5</td>
            <td>0.5</td>
          </tr>
          <tr>
            <td>2</td>
            <td>20</td>
            <td>22</td>
            <td>2</td>
          </tr>
          <tr>
            <td>3</td>
            <td>30</td>
            <td>33</td>
            <td>3</td>
          </tr>
        </tbody>
      </table> */}