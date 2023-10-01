OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
rx(-3.1033663749694824) q[0];
ry(-2.637911081314087) q[0];
rz(0.07993306964635849) q[0];
rx(-3.151022434234619) q[1];
ry(-3.1476480960845947) q[1];
rz(3.5850605964660645) q[1];
rx(-6.112255573272705) q[2];
ry(-2.3847086429595947) q[2];
rz(2.975665807723999) q[2];
rx(-3.113292694091797) q[3];
ry(-0.13100165128707886) q[3];
rz(3.106203317642212) q[3];
rx(2.067880392074585) q[4];
ry(-2.995485782623291) q[4];
rz(0.9450890421867371) q[4];
rx(3.141496419906616) q[5];
ry(-3.1416015625) q[5];
rz(-5.089786052703857) q[5];
rx(-2.1078920364379883) q[6];
ry(-1.298386812210083) q[6];
rz(2.7376837730407715) q[6];
rx(0.08981950581073761) q[7];
ry(-3.1568872928619385) q[7];
rz(-3.757394313812256) q[7];
rx(-4.908802509307861) q[8];
ry(-5.523519515991211) q[8];
rz(-3.7850189208984375) q[8];
rx(1.3770090341567993) q[9];
ry(0.28555765748023987) q[9];
rz(3.234581470489502) q[9];
rx(0.28563132882118225) q[10];
ry(3.2107207775115967) q[10];
rz(-0.3683367967605591) q[10];
rx(-2.122573137283325) q[11];
ry(2.0843019485473633) q[11];
rz(-2.097837209701538) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(-2.7186667919158936) q[0];
ry(-2.3799703121185303) q[0];
rz(-2.7861380577087402) q[0];
rx(-6.133353233337402) q[1];
ry(-3.7606260776519775) q[1];
rz(1.1207618713378906) q[1];
rx(-3.181825876235962) q[2];
ry(-3.9394631385803223) q[2];
rz(3.671564817428589) q[2];
rx(-5.1868391036987305) q[3];
ry(6.091940402984619) q[3];
rz(0.934380829334259) q[3];
rx(1.6383788585662842) q[4];
ry(-2.3936965465545654) q[4];
rz(1.7188529968261719) q[4];
rx(-3.216564893722534) q[5];
ry(-3.1388792991638184) q[5];
rz(-2.1072885990142822) q[5];
rx(4.4852705001831055) q[6];
ry(-2.2293753623962402) q[6];
rz(2.518019437789917) q[6];
rx(3.0231120586395264) q[7];
ry(-3.2070388793945312) q[7];
rz(4.816826820373535) q[7];
rx(-1.3047651052474976) q[8];
ry(-2.717487096786499) q[8];
rz(0.35193774104118347) q[8];
rx(1.2144153118133545) q[9];
ry(0.6480646729469299) q[9];
rz(1.654495358467102) q[9];
rx(-3.0846400260925293) q[10];
ry(3.6276352405548096) q[10];
rz(0.5318584442138672) q[10];
rx(-3.2709600925445557) q[11];
ry(1.3081680536270142) q[11];
rz(3.0746564865112305) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-4.5184526443481445) q[0];
ry(-3.4887266159057617) q[0];
rz(1.3729629516601562) q[0];
rx(-4.951905727386475) q[1];
ry(1.7428535223007202) q[1];
rz(2.326317548751831) q[1];
rx(5.074889659881592) q[2];
ry(-2.1979713439941406) q[2];
rz(1.6896435022354126) q[2];
rx(-0.287604957818985) q[3];
ry(3.995924949645996) q[3];
rz(-0.8926206231117249) q[3];
rx(5.195300579071045) q[4];
ry(-1.8704720735549927) q[4];
rz(-1.8221267461776733) q[4];
rx(2.6985223293304443) q[5];
ry(-3.1529181003570557) q[5];
rz(-0.02639847993850708) q[5];
rx(6.2831244468688965) q[6];
ry(-6.283331394195557) q[6];
rz(0.6747809052467346) q[6];
rx(0.14633537828922272) q[7];
ry(-0.011012950912117958) q[7];
rz(-1.3285350799560547) q[7];
rx(2.846688747406006) q[8];
ry(3.391164541244507) q[8];
rz(3.633516788482666) q[8];
rx(2.187037467956543) q[9];
ry(-2.7237136363983154) q[9];
rz(2.280897378921509) q[9];
rx(0.2230534702539444) q[10];
ry(3.2905874252319336) q[10];
rz(1.848406195640564) q[10];
rx(-1.6726984977722168) q[11];
ry(2.0541560649871826) q[11];
rz(2.6726484298706055) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(-4.746966361999512) q[0];
ry(-2.0617804527282715) q[0];
rz(2.8867526054382324) q[0];
rx(-2.6663711071014404) q[1];
ry(4.3769612312316895) q[1];
rz(1.3609651327133179) q[1];
rx(5.004534721374512) q[2];
ry(-8.463493347167969) q[2];
rz(4.346149444580078) q[2];
rx(-3.4205965995788574) q[3];
ry(-3.4852685928344727) q[3];
rz(1.8724501132965088) q[3];
rx(3.107383966445923) q[4];
ry(-6.251862049102783) q[4];
rz(-0.5094047784805298) q[4];
rx(1.6938811540603638) q[5];
ry(-5.024868965148926) q[5];
rz(-4.108473777770996) q[5];
rx(-0.00010288781049894169) q[6];
ry(-3.141552686691284) q[6];
rz(4.844253063201904) q[6];
rx(-0.1247929260134697) q[7];
ry(3.035428285598755) q[7];
rz(-7.598748207092285) q[7];
rx(-6.558699607849121) q[8];
ry(-8.990150451660156) q[8];
rz(-8.520547866821289) q[8];
rx(0.5846209526062012) q[9];
ry(-1.4086143970489502) q[9];
rz(5.262121200561523) q[9];
rx(-4.6923394203186035) q[10];
ry(3.052476644515991) q[10];
rz(-1.1554995775222778) q[10];
rx(-1.520072102546692) q[11];
ry(3.519935131072998) q[11];
rz(2.6627285480499268) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-1.7687045335769653) q[0];
ry(1.6295201778411865) q[0];
rz(1.6134896278381348) q[0];
rx(-0.37002983689308167) q[1];
ry(0.8865363001823425) q[1];
rz(1.1266282796859741) q[1];
rx(-0.10442983359098434) q[2];
ry(-6.268677711486816) q[2];
rz(-1.1048649549484253) q[2];
rx(-1.7690991163253784) q[3];
ry(0.035632193088531494) q[3];
rz(3.5882740020751953) q[3];
rx(3.1529688835144043) q[4];
ry(6.272030830383301) q[4];
rz(-4.495815277099609) q[4];
rx(2.9621634483337402) q[5];
ry(3.0531163215637207) q[5];
rz(3.878614664077759) q[5];
rx(-1.9721164790098555e-05) q[6];
ry(-3.141726493835449) q[6];
rz(-1.5968040227890015) q[6];
rx(6.160965442657471) q[7];
ry(-2.404993772506714) q[7];
rz(-1.961281180381775) q[7];
rx(-0.6175592541694641) q[8];
ry(-8.13664436340332) q[8];
rz(-8.11553955078125) q[8];
rx(3.069756507873535) q[9];
ry(-0.6422716975212097) q[9];
rz(-2.5346081256866455) q[9];
rx(3.214704990386963) q[10];
ry(7.468486785888672) q[10];
rz(1.4015648365020752) q[10];
rx(-3.6216065883636475) q[11];
ry(2.211960792541504) q[11];
rz(-3.6884610652923584) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(0.25843873620033264) q[0];
ry(0.9146238565444946) q[0];
rz(1.8211952447891235) q[0];
rx(-3.2773211002349854) q[1];
ry(4.557936668395996) q[1];
rz(3.51025652885437) q[1];
rx(-5.091702461242676) q[2];
ry(2.6867353916168213) q[2];
rz(6.135682106018066) q[2];
rx(-3.6919331550598145) q[3];
ry(1.8599351644515991) q[3];
rz(1.5469374656677246) q[3];
rx(3.1409807205200195) q[4];
ry(-0.00040643574902787805) q[4];
rz(-3.6907265186309814) q[4];
rx(3.8968591690063477) q[5];
ry(-4.7726521492004395) q[5];
rz(3.88830828666687) q[5];
rx(-3.1414968967437744) q[6];
ry(-3.1415975093841553) q[6];
rz(-2.2744970321655273) q[6];
rx(1.582747220993042) q[7];
ry(-5.135512828826904) q[7];
rz(-2.12084698677063) q[7];
rx(-6.082364559173584) q[8];
ry(2.7551708221435547) q[8];
rz(0.29196086525917053) q[8];
rx(-0.18713252246379852) q[9];
ry(-3.269784450531006) q[9];
rz(1.2119168043136597) q[9];
rx(3.07004714012146) q[10];
ry(2.9576358795166016) q[10];
rz(-3.2003753185272217) q[10];
rx(-3.0947980880737305) q[11];
ry(1.0110303163528442) q[11];
rz(2.520416259765625) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-3.037055730819702) q[0];
ry(-0.6647135615348816) q[0];
rz(3.86352801322937) q[0];
rx(-3.6086719036102295) q[1];
ry(3.2648963928222656) q[1];
rz(-4.437915802001953) q[1];
rx(-3.237361192703247) q[2];
ry(2.219799280166626) q[2];
rz(1.9113309383392334) q[2];
rx(0.4200301170349121) q[3];
ry(6.047240734100342) q[3];
rz(3.655855178833008) q[3];
rx(-3.548649549484253) q[4];
ry(6.919471263885498) q[4];
rz(-4.556828498840332) q[4];
rx(1.6025056838989258) q[5];
ry(-3.37636661529541) q[5];
rz(4.962056636810303) q[5];
rx(-3.145904541015625) q[6];
ry(0.9274868965148926) q[6];
rz(-3.0295562744140625) q[6];
rx(2.8765103816986084) q[7];
ry(-2.633167266845703) q[7];
rz(-1.0858540534973145) q[7];
rx(-6.678223609924316) q[8];
ry(4.424483776092529) q[8];
rz(-3.811337947845459) q[8];
rx(-3.0677530765533447) q[9];
ry(-6.846640110015869) q[9];
rz(3.739805221557617) q[9];
rx(-0.4316771626472473) q[10];
ry(4.54481315612793) q[10];
rz(5.510147571563721) q[10];
rx(-2.5712454319000244) q[11];
ry(0.17210420966148376) q[11];
rz(0.9552649855613708) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(1.9382591247558594) q[0];
ry(-4.442906379699707) q[0];
rz(7.004012107849121) q[0];
rx(-2.492398262023926) q[1];
ry(-1.524531364440918) q[1];
rz(-5.460372447967529) q[1];
rx(-2.1173255443573) q[2];
ry(-3.244987964630127) q[2];
rz(-1.9234044551849365) q[2];
rx(-3.455764055252075) q[3];
ry(0.6136588454246521) q[3];
rz(3.8344197273254395) q[3];
rx(-1.318685531616211) q[4];
ry(-0.25087466835975647) q[4];
rz(-3.6273105144500732) q[4];
rx(3.1414289474487305) q[5];
ry(-3.141496181488037) q[5];
rz(2.569533348083496) q[5];
rx(-3.2964625358581543) q[6];
ry(-4.059875011444092) q[6];
rz(6.595067501068115) q[6];
rx(2.2292604446411133) q[7];
ry(-0.138335183262825) q[7];
rz(0.925390362739563) q[7];
rx(-0.3701542615890503) q[8];
ry(2.8737196922302246) q[8];
rz(-1.9753265380859375) q[8];
rx(-3.117696762084961) q[9];
ry(-3.3915045261383057) q[9];
rz(0.9274415373802185) q[9];
rx(-2.6917479038238525) q[10];
ry(3.1037893295288086) q[10];
rz(5.340577602386475) q[10];
rx(-3.371284246444702) q[11];
ry(0.5514613389968872) q[11];
rz(3.593165636062622) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(0.6601449251174927) q[0];
ry(-3.880495548248291) q[0];
rz(0.1399049609899521) q[0];
rx(-0.12243351340293884) q[1];
ry(-0.8722871541976929) q[1];
rz(-5.894220352172852) q[1];
rx(-1.2065306901931763) q[2];
ry(-0.8815163373947144) q[2];
rz(-0.3228398263454437) q[2];
rx(0.13661664724349976) q[3];
ry(3.097356081008911) q[3];
rz(3.9662046432495117) q[3];
rx(-0.9128077030181885) q[4];
ry(2.0376393795013428) q[4];
rz(-4.7564778327941895) q[4];
rx(4.961493968963623) q[5];
ry(4.195478916168213) q[5];
rz(3.1205408573150635) q[5];
rx(3.141606092453003) q[6];
ry(-6.283100128173828) q[6];
rz(3.7544679641723633) q[6];
rx(6.088260173797607) q[7];
ry(-2.9122016429901123) q[7];
rz(3.3127267360687256) q[7];
rx(-2.6941704750061035) q[8];
ry(0.28894326090812683) q[8];
rz(-1.8272719383239746) q[8];
rx(2.750094413757324) q[9];
ry(-1.981976866722107) q[9];
rz(5.848913669586182) q[9];
rx(1.5018681287765503) q[10];
ry(-0.38316798210144043) q[10];
rz(3.470025062561035) q[10];
rx(-1.4946882724761963) q[11];
ry(-0.6942394971847534) q[11];
rz(1.5072448253631592) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(-3.8831863403320312) q[0];
ry(0.2595663070678711) q[0];
rz(-4.670489311218262) q[0];
rx(-7.846259593963623) q[1];
ry(0.02783200889825821) q[1];
rz(0.7642648220062256) q[1];
rx(-6.164814472198486) q[2];
ry(-0.023470617830753326) q[2];
rz(2.3017611503601074) q[2];
rx(-6.373136043548584) q[3];
ry(5.841663360595703) q[3];
rz(2.397909641265869) q[3];
rx(1.0668127536773682) q[4];
ry(-3.911926507949829) q[4];
rz(-3.5551695823669434) q[4];
rx(-1.4462960958480835) q[5];
ry(-0.19413051009178162) q[5];
rz(4.480454444885254) q[5];
rx(1.5763009786605835) q[6];
ry(-1.3621480464935303) q[6];
rz(-1.2907674312591553) q[6];
rx(2.9981229305267334) q[7];
ry(2.2988619804382324) q[7];
rz(-2.3387715816497803) q[7];
rx(-4.087372303009033) q[8];
ry(-1.0186184644699097) q[8];
rz(-2.401973009109497) q[8];
rx(-6.997167110443115) q[9];
ry(-4.081010818481445) q[9];
rz(0.6313743591308594) q[9];
rx(6.084840297698975) q[10];
ry(3.9398434162139893) q[10];
rz(3.030090093612671) q[10];
rx(-6.03495979309082) q[11];
ry(1.5103658437728882) q[11];
rz(5.297579288482666) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-5.152655601501465) q[0];
ry(2.3848562240600586) q[0];
rz(-3.170469045639038) q[0];
rx(-4.3170599937438965) q[1];
ry(5.492462158203125) q[1];
rz(1.1837068796157837) q[1];
rx(0.5119715929031372) q[2];
ry(3.8265247344970703) q[2];
rz(1.6937216520309448) q[2];
rx(0.12828607857227325) q[3];
ry(0.27580395340919495) q[3];
rz(3.397456169128418) q[3];
rx(5.958495616912842) q[4];
ry(3.157573938369751) q[4];
rz(-4.645758152008057) q[4];
rx(-3.141345500946045) q[5];
ry(-3.1414601802825928) q[5];
rz(4.183089256286621) q[5];
rx(-0.9730994701385498) q[6];
ry(-5.913914203643799) q[6];
rz(3.7017314434051514) q[6];
rx(-6.176244642119855e-05) q[7];
ry(-3.144747018814087) q[7];
rz(2.1709587574005127) q[7];
rx(3.9997193813323975) q[8];
ry(-0.906046450138092) q[8];
rz(-4.103850364685059) q[8];
rx(3.1381630897521973) q[9];
ry(-0.3258579969406128) q[9];
rz(1.3664361238479614) q[9];
rx(2.9099624156951904) q[10];
ry(-2.2001686096191406) q[10];
rz(0.6204266548156738) q[10];
rx(-1.5682129859924316) q[11];
ry(1.7813485860824585) q[11];
rz(1.5157922506332397) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(-0.8314509987831116) q[0];
ry(1.2811627388000488) q[0];
rz(-3.0962350368499756) q[0];
rx(-4.7412919998168945) q[1];
ry(-0.8535914421081543) q[1];
rz(3.1597626209259033) q[1];
rx(-2.33367919921875) q[2];
ry(-6.368030071258545) q[2];
rz(2.8845322132110596) q[2];
rx(2.007598400115967) q[3];
ry(3.198542594909668) q[3];
rz(0.4838450849056244) q[3];
rx(2.7281949520111084) q[4];
ry(3.113628387451172) q[4];
rz(-4.439859867095947) q[4];
rx(-3.1415460109710693) q[5];
ry(3.141597270965576) q[5];
rz(-5.0913004875183105) q[5];
rx(8.641382217407227) q[6];
ry(-0.6470195651054382) q[6];
rz(3.106254816055298) q[6];
rx(-2.6333398818969727) q[7];
ry(-7.855003356933594) q[7];
rz(2.915377378463745) q[7];
rx(-3.772656202316284) q[8];
ry(-1.5573413372039795) q[8];
rz(0.47458356618881226) q[8];
rx(3.1480302810668945) q[9];
ry(-3.0996365547180176) q[9];
rz(2.2098233699798584) q[9];
rx(2.4441516399383545) q[10];
ry(-0.6391101479530334) q[10];
rz(5.046354293823242) q[10];
rx(1.5825257301330566) q[11];
ry(1.5132038593292236) q[11];
rz(1.6686673164367676) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-1.847468614578247) q[0];
ry(-3.9034359455108643) q[0];
rz(-4.09201192855835) q[0];
rx(-4.342268466949463) q[1];
ry(6.596954822540283) q[1];
rz(2.828923463821411) q[1];
rx(-2.038229465484619) q[2];
ry(4.122170448303223) q[2];
rz(4.733750820159912) q[2];
rx(-5.948300361633301) q[3];
ry(-3.0025057792663574) q[3];
rz(4.966989040374756) q[3];
rx(0.7079514265060425) q[4];
ry(-0.4027981460094452) q[4];
rz(1.5556977987289429) q[4];
rx(-2.755186080932617) q[5];
ry(1.8468053340911865) q[5];
rz(-3.100229501724243) q[5];
rx(1.8512262105941772) q[6];
ry(-6.918811321258545) q[6];
rz(-2.2242627143859863) q[6];
rx(-2.9763991832733154) q[7];
ry(-2.618792772293091) q[7];
rz(7.806764602661133) q[7];
rx(-1.827820897102356) q[8];
ry(-1.4941844940185547) q[8];
rz(-2.322554588317871) q[8];
rx(1.7151131629943848) q[9];
ry(-1.0856002569198608) q[9];
rz(-4.783905982971191) q[9];
rx(0.46259447932243347) q[10];
ry(2.900097370147705) q[10];
rz(3.940708875656128) q[10];
rx(-2.4007372856140137) q[11];
ry(2.6055963039398193) q[11];
rz(0.8980275988578796) q[11];
rx(1.6510062217712402) q[0];
ry(-1.2681584358215332) q[0];
rz(0.4559739828109741) q[0];
rx(-5.795767307281494) q[1];
ry(3.3149499893188477) q[1];
rz(-0.6853182911872864) q[1];
rx(-2.2830142974853516) q[2];
ry(4.131401062011719) q[2];
rz(2.5629234313964844) q[2];
rx(0.23388424515724182) q[3];
ry(2.6138856410980225) q[3];
rz(-0.21155235171318054) q[3];
rx(-2.246436834335327) q[4];
ry(-5.55217170715332) q[4];
rz(3.7087221145629883) q[4];
rx(-4.0731658935546875) q[5];
ry(4.275426864624023) q[5];
rz(1.7809782028198242) q[5];
rx(5.5294389724731445) q[6];
ry(-2.2837371826171875) q[6];
rz(6.471372127532959) q[6];
rx(-1.0077086687088013) q[7];
ry(-0.23869109153747559) q[7];
rz(-1.0867936611175537) q[7];
rx(-3.111874580383301) q[8];
ry(0.07426822185516357) q[8];
rz(1.5173968076705933) q[8];
rx(4.410165309906006) q[9];
ry(1.493716835975647) q[9];
rz(-10.154967308044434) q[9];
rx(4.202603816986084) q[10];
ry(1.7288103103637695) q[10];
rz(-2.690340042114258) q[10];
rx(-2.945781707763672) q[11];
ry(2.274030923843384) q[11];
rz(-0.9449321627616882) q[11];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
cz q[3],q[4];
cz q[4],q[5];
cz q[5],q[6];
cz q[6],q[7];
cz q[7],q[8];
cz q[8],q[9];
cz q[9],q[10];
cz q[10],q[11];
cz q[11],q[0];
cz q[0],q[2];
cz q[1],q[3];
cz q[2],q[4];
cz q[3],q[5];
cz q[4],q[6];
cz q[5],q[7];
cz q[6],q[8];
cz q[7],q[9];
cz q[8],q[10];
cz q[9],q[11];
cz q[10],q[0];
cz q[11],q[1];
cz q[0],q[3];
cz q[1],q[4];
cz q[2],q[5];
cz q[3],q[6];
cz q[4],q[7];
cz q[5],q[8];
cz q[6],q[9];
cz q[7],q[10];
cz q[8],q[11];
cz q[9],q[0];
cz q[10],q[1];
cz q[11],q[2];
rx(3.0813801288604736) q[0];
ry(0.1861623227596283) q[0];
rz(0.2862032949924469) q[0];
rx(-3.1406824588775635) q[1];
ry(-0.01663372665643692) q[1];
rz(4.796666145324707) q[1];
rx(3.7335119247436523) q[2];
ry(-4.649523735046387) q[2];
rz(-4.672092914581299) q[2];
rx(3.1441471576690674) q[3];
ry(-3.1412813663482666) q[3];
rz(5.553987503051758) q[3];
rx(3.1434524059295654) q[4];
ry(3.1399550437927246) q[4];
rz(-1.8568427562713623) q[4];
rx(-6.646872407145565e-06) q[5];
ry(-4.814652493223548e-05) q[5];
rz(-1.654823660850525) q[5];
rx(0.00014239702431950718) q[6];
ry(-6.2830939292907715) q[6];
rz(2.44817852973938) q[6];
rx(-3.14166522026062) q[7];
ry(9.960110037354752e-05) q[7];
rz(3.6353771686553955) q[7];
rx(7.485891273972811e-06) q[8];
ry(-3.1415822505950928) q[8];
rz(2.753596067428589) q[8];
rx(3.1409993171691895) q[9];
ry(-3.1417410373687744) q[9];
rz(-6.822828769683838) q[9];
rx(-1.5834883015486412e-05) q[10];
ry(3.141589403152466) q[10];
rz(3.355022430419922) q[10];
rx(3.1415929794311523) q[11];
ry(-3.1415927410125732) q[11];
rz(-4.508528709411621) q[11];
