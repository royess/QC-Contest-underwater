OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
rx(-1.3902744054794312) q[0];
ry(-0.3457820415496826) q[0];
rz(1.3367303609848022) q[0];
rx(-3.060480833053589) q[1];
ry(-1.2921562194824219) q[1];
rz(0.2739720642566681) q[1];
rx(-5.551656723022461) q[2];
ry(0.269758939743042) q[2];
rz(0.921571671962738) q[2];
rx(-0.6088447570800781) q[3];
ry(1.0120397806167603) q[3];
rz(-4.695549488067627) q[3];
rx(-0.0006336732767522335) q[4];
ry(-3.141512632369995) q[4];
rz(-5.140448570251465) q[4];
rx(1.146056890487671) q[5];
ry(-0.9547616839408875) q[5];
rz(-1.552179217338562) q[5];
rx(-5.455020904541016) q[6];
ry(-6.051501750946045) q[6];
rz(1.471999168395996) q[6];
rx(-4.23585319519043) q[7];
ry(-1.918562412261963) q[7];
rz(1.5210827589035034) q[7];
rx(-1.5267038345336914) q[8];
ry(-5.260807514190674) q[8];
rz(-5.0115838050842285) q[8];
rx(-0.029258733615279198) q[9];
ry(-3.1338961124420166) q[9];
rz(-0.6620269417762756) q[9];
rx(-4.295690059661865) q[10];
ry(-0.8005560040473938) q[10];
rz(-2.151419162750244) q[10];
rx(2.267951250076294) q[11];
ry(2.99184513092041) q[11];
rz(1.6404632329940796) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(0.7875138521194458) q[0];
ry(1.3291676044464111) q[0];
rz(0.131709024310112) q[0];
rx(-2.1030616760253906) q[1];
ry(-1.3690248727798462) q[1];
rz(1.9231659173965454) q[1];
rx(-2.275357961654663) q[2];
ry(3.002633810043335) q[2];
rz(2.9731242656707764) q[2];
rx(-4.8692426681518555) q[3];
ry(1.5493429899215698) q[3];
rz(-0.7978992462158203) q[3];
rx(1.554642677307129) q[4];
ry(-2.8841073513031006) q[4];
rz(-1.568199634552002) q[4];
rx(-2.0853230953216553) q[5];
ry(-0.7137858271598816) q[5];
rz(-0.24531416594982147) q[5];
rx(0.8108521103858948) q[6];
ry(-2.573761224746704) q[6];
rz(3.296653985977173) q[6];
rx(-4.309661865234375) q[7];
ry(0.1033291444182396) q[7];
rz(3.386859655380249) q[7];
rx(0.5736868381500244) q[8];
ry(1.7885267734527588) q[8];
rz(-1.1650265455245972) q[8];
rx(5.947852611541748) q[9];
ry(-3.4229917526245117) q[9];
rz(0.7426975965499878) q[9];
rx(-0.8472265005111694) q[10];
ry(3.810007333755493) q[10];
rz(1.4180660247802734) q[10];
rx(3.075038194656372) q[11];
ry(4.022582530975342) q[11];
rz(5.598303318023682) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-2.8206024169921875) q[0];
ry(-0.8698710799217224) q[0];
rz(5.864120006561279) q[0];
rx(-3.198546886444092) q[1];
ry(0.029829181730747223) q[1];
rz(3.2850661277770996) q[1];
rx(-0.00037172817974351346) q[2];
ry(3.143796443939209) q[2];
rz(1.2350332736968994) q[2];
rx(-1.7687259912490845) q[3];
ry(2.4044389724731445) q[3];
rz(0.5564455389976501) q[3];
rx(3.123660087585449) q[4];
ry(-1.5655298233032227) q[4];
rz(-0.8336498141288757) q[4];
rx(3.1438400745391846) q[5];
ry(-3.1411969661712646) q[5];
rz(-1.2419015169143677) q[5];
rx(-3.46817684173584) q[6];
ry(-3.224529504776001) q[6];
rz(2.8665778636932373) q[6];
rx(-0.00014934498176444322) q[7];
ry(0.0004867701791226864) q[7];
rz(-4.081809043884277) q[7];
rx(-3.0611932277679443) q[8];
ry(-3.2930068969726562) q[8];
rz(-3.456178665161133) q[8];
rx(0.8849897384643555) q[9];
ry(-0.33523449301719666) q[9];
rz(-7.679105758666992) q[9];
rx(-3.8775570392608643) q[10];
ry(-2.576169490814209) q[10];
rz(0.1955954134464264) q[10];
rx(0.9377259016036987) q[11];
ry(1.9241832494735718) q[11];
rz(0.833439290523529) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(-6.273128509521484) q[0];
ry(4.053979873657227) q[0];
rz(4.47467565536499) q[0];
rx(-2.2120442390441895) q[1];
ry(1.823624610900879) q[1];
rz(1.557194471359253) q[1];
rx(0.2277357280254364) q[2];
ry(0.6316590905189514) q[2];
rz(0.6749445199966431) q[2];
rx(-3.7362186908721924) q[3];
ry(-2.438734292984009) q[3];
rz(1.0119080543518066) q[3];
rx(3.1416447162628174) q[4];
ry(-3.1423113346099854) q[4];
rz(0.04598020017147064) q[4];
rx(1.577712059020996) q[5];
ry(-2.8761377334594727) q[5];
rz(-4.157859802246094) q[5];
rx(-0.20750094950199127) q[6];
ry(-3.3449819087982178) q[6];
rz(4.624967575073242) q[6];
rx(6.2786946296691895) q[7];
ry(3.1429245471954346) q[7];
rz(-2.3701095581054688) q[7];
rx(-0.365070641040802) q[8];
ry(-3.6136555671691895) q[8];
rz(5.001815319061279) q[8];
rx(3.331559181213379) q[9];
ry(-0.01274543721228838) q[9];
rz(0.5678043961524963) q[9];
rx(-2.0493886470794678) q[10];
ry(1.1315476894378662) q[10];
rz(-2.8665428161621094) q[10];
rx(3.7107744216918945) q[11];
ry(1.816339373588562) q[11];
rz(-0.7543563842773438) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(4.006646156311035) q[0];
ry(0.5518469214439392) q[0];
rz(4.303431510925293) q[0];
rx(-5.288730144500732) q[1];
ry(0.6936287879943848) q[1];
rz(6.41255521774292) q[1];
rx(-2.4525535106658936) q[2];
ry(0.9273267984390259) q[2];
rz(0.46873852610588074) q[2];
rx(-1.7811405658721924) q[3];
ry(-0.3424662947654724) q[3];
rz(1.9846258163452148) q[3];
rx(2.8064777851104736) q[4];
ry(2.587236166000366) q[4];
rz(-3.2196099758148193) q[4];
rx(2.943106174468994) q[5];
ry(0.124376580119133) q[5];
rz(0.612446665763855) q[5];
rx(-5.2745890617370605) q[6];
ry(2.1211843490600586) q[6];
rz(2.7385566234588623) q[6];
rx(-8.628228187561035) q[7];
ry(1.7582690715789795) q[7];
rz(1.6810407638549805) q[7];
rx(1.5134553909301758) q[8];
ry(1.9819976091384888) q[8];
rz(5.365874290466309) q[8];
rx(-2.1632840633392334) q[9];
ry(-1.6546984910964966) q[9];
rz(-1.8817918300628662) q[9];
rx(1.6145952939987183) q[10];
ry(1.4761055707931519) q[10];
rz(-2.897191286087036) q[10];
rx(-3.2679264545440674) q[11];
ry(1.3405934572219849) q[11];
rz(-0.09972038120031357) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(-3.157951831817627) q[0];
ry(2.156684637069702) q[0];
rz(1.255245327949524) q[0];
rx(-3.8927242755889893) q[1];
ry(-0.25188907980918884) q[1];
rz(0.012523471377789974) q[1];
rx(-3.141077995300293) q[2];
ry(3.1434378623962402) q[2];
rz(1.248763084411621) q[2];
rx(-1.031959056854248) q[3];
ry(2.2573163509368896) q[3];
rz(3.8699910640716553) q[3];
rx(2.6249496936798096) q[4];
ry(-2.7352242469787598) q[4];
rz(-2.0803229808807373) q[4];
rx(1.8000154495239258) q[5];
ry(-4.6407904624938965) q[5];
rz(1.0102369785308838) q[5];
rx(-3.142179489135742) q[6];
ry(-0.0017348862020298839) q[6];
rz(4.674198627471924) q[6];
rx(2.6796042919158936) q[7];
ry(-3.7001399993896484) q[7];
rz(0.32278361916542053) q[7];
rx(-3.181887149810791) q[8];
ry(7.856211185455322) q[8];
rz(2.553151845932007) q[8];
rx(3.4720094203948975) q[9];
ry(3.523385763168335) q[9];
rz(-1.418623447418213) q[9];
rx(3.816228151321411) q[10];
ry(0.1439075767993927) q[10];
rz(-1.181746006011963) q[10];
rx(0.1286451518535614) q[11];
ry(1.80154287815094) q[11];
rz(-3.0773768424987793) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(2.5160012245178223) q[0];
ry(2.0131731033325195) q[0];
rz(0.2024681121110916) q[0];
rx(-3.8803703784942627) q[1];
ry(-0.07823186367750168) q[1];
rz(-1.3933547735214233) q[1];
rx(0.8487106561660767) q[2];
ry(1.5639393329620361) q[2];
rz(0.42707428336143494) q[2];
rx(-3.5875403881073) q[3];
ry(3.2920010089874268) q[3];
rz(1.4999139308929443) q[3];
rx(2.646214485168457) q[4];
ry(-4.893306732177734) q[4];
rz(0.5524706244468689) q[4];
rx(1.6602673530578613) q[5];
ry(-1.7906264066696167) q[5];
rz(0.054937880486249924) q[5];
rx(-3.1415114402770996) q[6];
ry(-7.117594941519201e-05) q[6];
rz(1.2008076906204224) q[6];
rx(4.045806407928467) q[7];
ry(-3.9164249897003174) q[7];
rz(4.644446849822998) q[7];
rx(-0.12259483337402344) q[8];
ry(5.221274375915527) q[8];
rz(-0.8532277345657349) q[8];
rx(-3.5846025943756104) q[9];
ry(0.32257401943206787) q[9];
rz(2.231168270111084) q[9];
rx(-2.7184839248657227) q[10];
ry(2.5450592041015625) q[10];
rz(0.3502075970172882) q[10];
rx(-0.14152558147907257) q[11];
ry(2.3907952308654785) q[11];
rz(-4.844004154205322) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(0.5846337080001831) q[0];
ry(3.3205859661102295) q[0];
rz(-1.98188054561615) q[0];
rx(-1.5739142894744873) q[1];
ry(6.018866062164307) q[1];
rz(-0.9418132305145264) q[1];
rx(-4.703548431396484) q[2];
ry(-0.4107503890991211) q[2];
rz(0.5799107551574707) q[2];
rx(-5.928678512573242) q[3];
ry(3.2744998931884766) q[3];
rz(0.16420230269432068) q[3];
rx(1.7608674764633179) q[4];
ry(-8.995681762695312) q[4];
rz(-2.8768529891967773) q[4];
rx(2.265861749649048) q[5];
ry(-4.527748107910156) q[5];
rz(1.6254496574401855) q[5];
rx(-3.1418261528015137) q[6];
ry(-3.1414785385131836) q[6];
rz(3.781907558441162) q[6];
rx(1.8477617502212524) q[7];
ry(-2.6569838523864746) q[7];
rz(0.823828935623169) q[7];
rx(-6.591508388519287) q[8];
ry(6.415989398956299) q[8];
rz(-3.9151108264923096) q[8];
rx(-3.6591343879699707) q[9];
ry(0.03360724076628685) q[9];
rz(2.7924997806549072) q[9];
rx(-2.172348976135254) q[10];
ry(1.3210258483886719) q[10];
rz(-0.6116227507591248) q[10];
rx(-3.901339530944824) q[11];
ry(0.05134810134768486) q[11];
rz(-1.8283114433288574) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-2.79193115234375) q[0];
ry(-6.282131671905518) q[0];
rz(0.5357335805892944) q[0];
rx(-3.1416027545928955) q[1];
ry(3.1415886878967285) q[1];
rz(-1.100362777709961) q[1];
rx(-4.359195709228516) q[2];
ry(1.5436433553695679) q[2];
rz(-1.2285281419754028) q[2];
rx(-3.520585536956787) q[3];
ry(2.8583593368530273) q[3];
rz(-2.6881961822509766) q[3];
rx(3.1419363021850586) q[4];
ry(-0.001256604795344174) q[4];
rz(-2.754375457763672) q[4];
rx(-1.568067193031311) q[5];
ry(0.6309986710548401) q[5];
rz(4.8401665687561035) q[5];
rx(-0.000714011664967984) q[6];
ry(-0.0008666784269735217) q[6];
rz(-0.18859626352787018) q[6];
rx(-4.092246055603027) q[7];
ry(-4.617035388946533) q[7];
rz(-4.444151878356934) q[7];
rx(-5.648795127868652) q[8];
ry(-5.807956218719482) q[8];
rz(5.429330825805664) q[8];
rx(3.612337827682495) q[9];
ry(-3.172893762588501) q[9];
rz(1.8759177923202515) q[9];
rx(4.102792263031006) q[10];
ry(2.061246871948242) q[10];
rz(3.289365291595459) q[10];
rx(-4.607461929321289) q[11];
ry(2.244581937789917) q[11];
rz(1.8519805669784546) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(-4.439581871032715) q[0];
ry(-1.2543461322784424) q[0];
rz(-4.030865669250488) q[0];
rx(-2.855731725692749) q[1];
ry(4.712450981140137) q[1];
rz(-1.2417000532150269) q[1];
rx(5.283755302429199) q[2];
ry(1.585902214050293) q[2];
rz(-6.182541370391846) q[2];
rx(0.4389769732952118) q[3];
ry(3.2983126640319824) q[3];
rz(4.078872203826904) q[3];
rx(3.142517328262329) q[4];
ry(-3.1405320167541504) q[4];
rz(-1.9716532230377197) q[4];
rx(-4.72974967956543) q[5];
ry(-3.261052131652832) q[5];
rz(2.23343825340271) q[5];
rx(-1.1237120628356934) q[6];
ry(1.8087414503097534) q[6];
rz(1.056658148765564) q[6];
rx(-0.7099300622940063) q[7];
ry(2.1856331825256348) q[7];
rz(-6.392641544342041) q[7];
rx(-5.368034839630127) q[8];
ry(1.9685982465744019) q[8];
rz(3.7808806896209717) q[8];
rx(3.7232534885406494) q[9];
ry(-1.4375759363174438) q[9];
rz(2.0842978954315186) q[9];
rx(3.12611722946167) q[10];
ry(0.07793112844228745) q[10];
rz(2.4725747108459473) q[10];
rx(1.2521533966064453) q[11];
ry(1.7803677320480347) q[11];
rz(-3.1170694828033447) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(4.409480094909668) q[0];
ry(1.4159611463546753) q[0];
rz(-2.3477466106414795) q[0];
rx(-4.528085708618164) q[1];
ry(-0.6520802974700928) q[1];
rz(-3.2634477615356445) q[1];
rx(2.244131088256836) q[2];
ry(7.951993465423584) q[2];
rz(-4.000866413116455) q[2];
rx(1.6449042558670044) q[3];
ry(1.5740940570831299) q[3];
rz(2.1204545497894287) q[3];
rx(2.6354475021362305) q[4];
ry(3.714808464050293) q[4];
rz(-2.073826313018799) q[4];
rx(-3.1078217029571533) q[5];
ry(-3.9266412258148193) q[5];
rz(1.6355184316635132) q[5];
rx(0.8652624487876892) q[6];
ry(0.8991038203239441) q[6];
rz(-0.19752174615859985) q[6];
rx(-2.2127275466918945) q[7];
ry(-2.4876644611358643) q[7];
rz(-3.998255729675293) q[7];
rx(-0.9528051018714905) q[8];
ry(-1.112575650215149) q[8];
rz(1.7277586460113525) q[8];
rx(2.4044697284698486) q[9];
ry(-2.2775323390960693) q[9];
rz(0.29251816868782043) q[9];
rx(4.798729419708252) q[10];
ry(5.113826274871826) q[10];
rz(-1.6304446458816528) q[10];
rx(-1.5767894983291626) q[11];
ry(2.543213129043579) q[11];
rz(0.797274112701416) q[11];
rx(1.4522147178649902) q[0];
ry(0.28591519594192505) q[0];
rz(-1.3929111957550049) q[0];
rx(-3.025438070297241) q[1];
ry(-1.8651726245880127) q[1];
rz(-8.386180877685547) q[1];
rx(6.983953952789307) q[2];
ry(-1.4950337409973145) q[2];
rz(11.7299165725708) q[2];
rx(-5.111051082611084) q[3];
ry(1.9165878295898438) q[3];
rz(-1.289986491203308) q[3];
rx(3.1710448265075684) q[4];
ry(5.641570568084717) q[4];
rz(1.4971504211425781) q[4];
rx(-2.3624300956726074) q[5];
ry(-3.847161293029785) q[5];
rz(-1.7582465410232544) q[5];
rx(6.91108512878418) q[6];
ry(2.9531502723693848) q[6];
rz(-8.506912231445312) q[6];
rx(-3.6305296421051025) q[7];
ry(-7.286966800689697) q[7];
rz(4.428853511810303) q[7];
rx(1.9426543712615967) q[8];
ry(-0.7988835573196411) q[8];
rz(5.676686763763428) q[8];
rx(1.9290555715560913) q[9];
ry(-2.635504722595215) q[9];
rz(7.2945661544799805) q[9];
rx(2.31982684135437) q[10];
ry(3.978182554244995) q[10];
rz(-3.2750654220581055) q[10];
rx(-1.5779727697372437) q[11];
ry(-0.8012265563011169) q[11];
rz(9.617137908935547) q[11];
