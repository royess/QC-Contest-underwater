OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
rx(-3.972543239593506) q[0];
ry(1.2695598602294922) q[0];
rz(4.724609851837158) q[0];
rx(-1.9199000597000122) q[1];
ry(-4.563100337982178) q[1];
rz(1.127414584159851) q[1];
rx(0.019717903807759285) q[2];
ry(-3.3363142013549805) q[2];
rz(-3.0509753227233887) q[2];
rx(-4.2910075187683105) q[3];
ry(5.293426513671875) q[3];
rz(-2.8540050983428955) q[3];
rx(-0.08166778832674026) q[4];
ry(-2.94775128364563) q[4];
rz(6.855024814605713) q[4];
rx(4.738374710083008) q[5];
ry(-1.0437145233154297) q[5];
rz(-1.5618064403533936) q[5];
rx(-3.1422877311706543) q[6];
ry(-3.140000581741333) q[6];
rz(3.9896748065948486) q[6];
rx(3.210421085357666) q[7];
ry(3.176640272140503) q[7];
rz(0.9913742542266846) q[7];
rx(-2.1937143802642822) q[8];
ry(-3.535688638687134) q[8];
rz(2.356792688369751) q[8];
rx(-3.429194927215576) q[9];
ry(6.065027713775635) q[9];
rz(-0.4005168676376343) q[9];
rx(-1.5022639036178589) q[10];
ry(5.591289043426514) q[10];
rz(-2.097604751586914) q[10];
rx(4.473409652709961) q[11];
ry(3.5486676692962646) q[11];
rz(1.8808118104934692) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(-5.408968448638916) q[0];
ry(-1.5210479497909546) q[0];
rz(-7.0203776359558105) q[0];
rx(-1.858650803565979) q[1];
ry(-1.4252326488494873) q[1];
rz(1.9591090679168701) q[1];
rx(-6.2890496253967285) q[2];
ry(0.2062539905309677) q[2];
rz(-2.4414138793945312) q[2];
rx(-4.947790145874023) q[3];
ry(0.5843822360038757) q[3];
rz(-2.298689842224121) q[3];
rx(-3.1505651473999023) q[4];
ry(-3.194650411605835) q[4];
rz(3.680995464324951) q[4];
rx(4.2256669998168945) q[5];
ry(-4.751200199127197) q[5];
rz(-0.5985206365585327) q[5];
rx(1.4225451946258545) q[6];
ry(-4.595262050628662) q[6];
rz(0.6876091957092285) q[6];
rx(3.4166157245635986) q[7];
ry(0.18581555783748627) q[7];
rz(2.965820550918579) q[7];
rx(-5.492637634277344) q[8];
ry(-0.21804071962833405) q[8];
rz(-4.7109832763671875) q[8];
rx(0.2359813004732132) q[9];
ry(6.470363616943359) q[9];
rz(-4.620136737823486) q[9];
rx(-2.1238439083099365) q[10];
ry(-0.11234936118125916) q[10];
rz(6.761324405670166) q[10];
rx(2.37985897064209) q[11];
ry(5.017929553985596) q[11];
rz(2.8016529083251953) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-1.0562586784362793) q[0];
ry(-3.2789669036865234) q[0];
rz(1.8330367803573608) q[0];
rx(-3.195387363433838) q[1];
ry(-3.449051856994629) q[1];
rz(-1.0931732654571533) q[1];
rx(-0.9736824631690979) q[2];
ry(-1.1831523180007935) q[2];
rz(-3.2442879676818848) q[2];
rx(-3.1415984630584717) q[3];
ry(3.1415388584136963) q[3];
rz(-1.350796103477478) q[3];
rx(1.2815775871276855) q[4];
ry(-3.0694613456726074) q[4];
rz(2.9554882049560547) q[4];
rx(3.255779504776001) q[5];
ry(-4.03354549407959) q[5];
rz(0.378893107175827) q[5];
rx(1.8620920181274414) q[6];
ry(-6.2664995193481445) q[6];
rz(4.907503604888916) q[6];
rx(0.15510526299476624) q[7];
ry(-6.335824966430664) q[7];
rz(-8.773807525634766) q[7];
rx(0.09565062820911407) q[8];
ry(-2.013727903366089) q[8];
rz(3.5020511150360107) q[8];
rx(-3.377737522125244) q[9];
ry(-1.6146240234375) q[9];
rz(3.1880199909210205) q[9];
rx(-2.277758836746216) q[10];
ry(1.2326232194900513) q[10];
rz(1.9745128154754639) q[10];
rx(3.6965906620025635) q[11];
ry(-0.6499783396720886) q[11];
rz(0.7710080742835999) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(-4.282702922821045) q[0];
ry(0.2660311758518219) q[0];
rz(-0.9727943539619446) q[0];
rx(6.894572734832764) q[1];
ry(0.14840546250343323) q[1];
rz(-1.4671508073806763) q[1];
rx(1.9598872661590576) q[2];
ry(0.6658821702003479) q[2];
rz(-1.3809632062911987) q[2];
rx(-5.448659896850586) q[3];
ry(1.5977798700332642) q[3];
rz(5.248398303985596) q[3];
rx(-1.855249047279358) q[4];
ry(0.19454964995384216) q[4];
rz(-2.2660601139068604) q[4];
rx(2.329908609390259) q[5];
ry(-5.726794242858887) q[5];
rz(5.146949291229248) q[5];
rx(-5.4332685470581055) q[6];
ry(-3.7665581703186035) q[6];
rz(3.189741373062134) q[6];
rx(3.0424318313598633) q[7];
ry(5.928256034851074) q[7];
rz(-1.3696482181549072) q[7];
rx(2.9898927211761475) q[8];
ry(-3.0288312435150146) q[8];
rz(7.454753398895264) q[8];
rx(-0.7587892413139343) q[9];
ry(5.8004374504089355) q[9];
rz(7.817623138427734) q[9];
rx(-0.31703564524650574) q[10];
ry(8.08358383178711) q[10];
rz(-9.020256996154785) q[10];
rx(2.313613176345825) q[11];
ry(-0.0016587382415309548) q[11];
rz(-6.598480224609375) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-1.0795329809188843) q[0];
ry(0.14310906827449799) q[0];
rz(-4.373088836669922) q[0];
rx(6.320054531097412) q[1];
ry(-0.07826130837202072) q[1];
rz(6.4622802734375) q[1];
rx(8.031465530395508) q[2];
ry(-2.477248191833496) q[2];
rz(8.781993865966797) q[2];
rx(-2.4886856079101562) q[3];
ry(3.228935480117798) q[3];
rz(8.092611312866211) q[3];
rx(-3.1414456367492676) q[4];
ry(-3.1416988372802734) q[4];
rz(1.433001160621643) q[4];
rx(0.6144828796386719) q[5];
ry(-1.3809856176376343) q[5];
rz(2.19387149810791) q[5];
rx(4.978320598602295) q[6];
ry(-0.3893616795539856) q[6];
rz(3.085256338119507) q[6];
rx(1.2920613288879395) q[7];
ry(1.5942078828811646) q[7];
rz(-1.427880883216858) q[7];
rx(3.1690988540649414) q[8];
ry(-3.1516668796539307) q[8];
rz(0.8000153303146362) q[8];
rx(0.14059455692768097) q[9];
ry(-6.299241065979004) q[9];
rz(5.923665523529053) q[9];
rx(1.3148589134216309) q[10];
ry(7.282145977020264) q[10];
rz(-3.3392367362976074) q[10];
rx(4.858221054077148) q[11];
ry(4.832054615020752) q[11];
rz(-5.137630939483643) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(3.0258231163024902) q[0];
ry(2.1938440799713135) q[0];
rz(-0.09891515225172043) q[0];
rx(0.18186774849891663) q[1];
ry(3.3280868530273438) q[1];
rz(0.3932899832725525) q[1];
rx(-3.358341932296753) q[2];
ry(9.351880073547363) q[2];
rz(3.3452959060668945) q[2];
rx(-1.9499086141586304) q[3];
ry(8.28792953491211) q[3];
rz(0.4504382312297821) q[3];
rx(3.1414716243743896) q[4];
ry(-3.141937494277954) q[4];
rz(-3.2213497161865234) q[4];
rx(3.466804265975952) q[5];
ry(-6.883914947509766) q[5];
rz(3.270648241043091) q[5];
rx(0.36995506286621094) q[6];
ry(-2.5951266288757324) q[6];
rz(-0.09499258548021317) q[6];
rx(2.070034980773926) q[7];
ry(4.806488037109375) q[7];
rz(0.2615330219268799) q[7];
rx(0.009270003996789455) q[8];
ry(0.02760014683008194) q[8];
rz(-3.681933879852295) q[8];
rx(-6.0244646072387695) q[9];
ry(-3.414734363555908) q[9];
rz(-0.17608754336833954) q[9];
rx(-0.2976786196231842) q[10];
ry(5.407907485961914) q[10];
rz(-4.499807357788086) q[10];
rx(4.649932384490967) q[11];
ry(4.993480682373047) q[11];
rz(-3.5719361305236816) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-6.807999610900879) q[0];
ry(-4.287357330322266) q[0];
rz(2.634251832962036) q[0];
rx(0.509553849697113) q[1];
ry(-1.9707229137420654) q[1];
rz(8.871390342712402) q[1];
rx(-4.037430286407471) q[2];
ry(1.6440173387527466) q[2];
rz(-2.6803934574127197) q[2];
rx(-4.530072212219238) q[3];
ry(1.891920804977417) q[3];
rz(4.121096611022949) q[3];
rx(3.1416287422180176) q[4];
ry(-3.141516923904419) q[4];
rz(-1.3152865171432495) q[4];
rx(2.7677860260009766) q[5];
ry(-3.473167896270752) q[5];
rz(3.9307451248168945) q[5];
rx(-3.110868453979492) q[6];
ry(-6.376541614532471) q[6];
rz(5.481175899505615) q[6];
rx(4.266598701477051) q[7];
ry(-0.5843963027000427) q[7];
rz(-1.3490620851516724) q[7];
rx(0.039184942841529846) q[8];
ry(-9.353198051452637) q[8];
rz(-3.0461597442626953) q[8];
rx(-3.716728925704956) q[9];
ry(-2.259770393371582) q[9];
rz(1.90070641040802) q[9];
rx(0.8724239468574524) q[10];
ry(2.2062041759490967) q[10];
rz(-9.5282564163208) q[10];
rx(1.2857794761657715) q[11];
ry(4.929283618927002) q[11];
rz(-4.451220989227295) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(0.43375709652900696) q[0];
ry(-1.6341310739517212) q[0];
rz(2.478780508041382) q[0];
rx(-6.598905563354492) q[1];
ry(-0.8646020889282227) q[1];
rz(1.2750589847564697) q[1];
rx(2.1177151203155518) q[2];
ry(-1.859009861946106) q[2];
rz(-0.933140218257904) q[2];
rx(-2.0788168907165527) q[3];
ry(3.8564233779907227) q[3];
rz(0.6322336792945862) q[3];
rx(1.5708519220352173) q[4];
ry(-0.9089170098304749) q[4];
rz(-4.568704605102539) q[4];
rx(6.989372730255127) q[5];
ry(-2.58329439163208) q[5];
rz(3.9840896129608154) q[5];
rx(-2.7771432399749756) q[6];
ry(-3.51676869392395) q[6];
rz(2.735618829727173) q[6];
rx(4.827362060546875) q[7];
ry(1.9554808139801025) q[7];
rz(-3.322174310684204) q[7];
rx(-2.07551908493042) q[8];
ry(1.4449201822280884) q[8];
rz(-1.1956814527511597) q[8];
rx(-1.926073431968689) q[9];
ry(-0.23518683016300201) q[9];
rz(2.1540791988372803) q[9];
rx(1.3115290403366089) q[10];
ry(2.00386643409729) q[10];
rz(-4.397082328796387) q[10];
rx(5.9820332527160645) q[11];
ry(4.77547025680542) q[11];
rz(-2.7036683559417725) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-0.9187517166137695) q[0];
ry(1.5652718544006348) q[0];
rz(-2.5454463958740234) q[0];
rx(6.546090126037598) q[1];
ry(9.670404434204102) q[1];
rz(5.949188709259033) q[1];
rx(-10.39062786102295) q[2];
ry(3.818188428878784) q[2];
rz(4.505552768707275) q[2];
rx(-4.9432783126831055) q[3];
ry(3.4638419151306152) q[3];
rz(3.0159056186676025) q[3];
rx(-7.855400085449219) q[4];
ry(1.4271055459976196) q[4];
rz(-4.827848434448242) q[4];
rx(-3.141984224319458) q[5];
ry(-3.141758918762207) q[5];
rz(5.391842842102051) q[5];
rx(-3.172853469848633) q[6];
ry(0.04809782654047012) q[6];
rz(4.16852331161499) q[6];
rx(3.4522149562835693) q[7];
ry(0.8109440207481384) q[7];
rz(-0.5622861385345459) q[7];
rx(-3.9265666007995605) q[8];
ry(6.985762596130371) q[8];
rz(1.8365811109542847) q[8];
rx(-0.5481892228126526) q[9];
ry(-0.7456564903259277) q[9];
rz(3.76979660987854) q[9];
rx(3.782747268676758) q[10];
ry(3.9890472888946533) q[10];
rz(3.494527816772461) q[10];
rx(4.622921466827393) q[11];
ry(3.8515663146972656) q[11];
rz(-4.584230422973633) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(-1.5991078615188599) q[0];
ry(-3.017909288406372) q[0];
rz(-1.1735796928405762) q[0];
rx(-0.1721394956111908) q[1];
ry(5.754333972930908) q[1];
rz(-3.8752713203430176) q[1];
rx(10.161581039428711) q[2];
ry(-1.029269814491272) q[2];
rz(-2.586581230163574) q[2];
rx(2.020885705947876) q[3];
ry(4.845416069030762) q[3];
rz(-0.3017592132091522) q[3];
rx(3.141585111618042) q[4];
ry(-3.1415936946868896) q[4];
rz(-4.126047611236572) q[4];
rx(-4.698052406311035) q[5];
ry(1.5088708400726318) q[5];
rz(3.968416213989258) q[5];
rx(-0.4568929076194763) q[6];
ry(-0.6157419085502625) q[6];
rz(6.12814998626709) q[6];
rx(10.195320129394531) q[7];
ry(2.9515535831451416) q[7];
rz(-7.01469612121582) q[7];
rx(-0.7444480657577515) q[8];
ry(-3.4490699768066406) q[8];
rz(-1.6648534536361694) q[8];
rx(1.8269010782241821) q[9];
ry(-6.57697868347168) q[9];
rz(3.543421745300293) q[9];
rx(4.096963405609131) q[10];
ry(4.8539719581604) q[10];
rz(6.380685806274414) q[10];
rx(5.497530937194824) q[11];
ry(1.4737818241119385) q[11];
rz(-5.327556133270264) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-2.8718960285186768) q[0];
ry(-0.8636401295661926) q[0];
rz(-1.2804675102233887) q[0];
rx(-1.909854531288147) q[1];
ry(-2.138218879699707) q[1];
rz(1.2574793100357056) q[1];
rx(4.715359210968018) q[2];
ry(6.414004325866699) q[2];
rz(-0.813501238822937) q[2];
rx(6.067682266235352) q[3];
ry(0.23994652926921844) q[3];
rz(-4.249444484710693) q[3];
rx(-3.142498016357422) q[4];
ry(3.1430439949035645) q[4];
rz(-3.8345627784729004) q[4];
rx(-4.713260650634766) q[5];
ry(0.8124611973762512) q[5];
rz(4.9561662673950195) q[5];
rx(-3.2316348552703857) q[6];
ry(0.12452773749828339) q[6];
rz(4.769245624542236) q[6];
rx(-3.4536075592041016) q[7];
ry(-4.060809135437012) q[7];
rz(-1.9923639297485352) q[7];
rx(0.17780931293964386) q[8];
ry(-6.474893093109131) q[8];
rz(-4.904567241668701) q[8];
rx(1.5982747077941895) q[9];
ry(-5.082042217254639) q[9];
rz(-0.47597092390060425) q[9];
rx(4.289163112640381) q[10];
ry(1.6411489248275757) q[10];
rz(0.3553265333175659) q[10];
rx(-0.5046671032905579) q[11];
ry(-3.0199787616729736) q[11];
rz(-2.8003461360931396) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(-0.3497864305973053) q[0];
ry(-2.6590099334716797) q[0];
rz(-0.13880380988121033) q[0];
rx(-7.914190769195557) q[1];
ry(-6.36334753036499) q[1];
rz(2.532097816467285) q[1];
rx(-3.399254322052002) q[2];
ry(-4.588769435882568) q[2];
rz(6.40266752243042) q[2];
rx(-0.6204034686088562) q[3];
ry(4.224048614501953) q[3];
rz(1.3565700054168701) q[3];
rx(4.282135963439941) q[4];
ry(-7.9714860916137695) q[4];
rz(5.107806205749512) q[4];
rx(-3.141507148742676) q[5];
ry(-6.283397674560547) q[5];
rz(2.7234060764312744) q[5];
rx(2.265965461730957) q[6];
ry(-2.1317665576934814) q[6];
rz(6.293779373168945) q[6];
rx(2.9918837547302246) q[7];
ry(-6.893133640289307) q[7];
rz(-4.698297023773193) q[7];
rx(-0.03264836594462395) q[8];
ry(-3.045041561126709) q[8];
rz(2.2837107181549072) q[8];
rx(5.499650478363037) q[9];
ry(3.68963885307312) q[9];
rz(-4.462940216064453) q[9];
rx(-3.9570810794830322) q[10];
ry(6.2129998207092285) q[10];
rz(3.367511510848999) q[10];
rx(-2.6919121742248535) q[11];
ry(-3.4074511528015137) q[11];
rz(-3.2439959049224854) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-5.429934501647949) q[0];
ry(1.6464481353759766) q[0];
rz(1.7481191158294678) q[0];
rx(1.6440317630767822) q[1];
ry(1.9787367582321167) q[1];
rz(-0.3053382933139801) q[1];
rx(1.8080997467041016) q[2];
ry(4.81461763381958) q[2];
rz(-1.7209993600845337) q[2];
rx(1.5075825452804565) q[3];
ry(2.1193184852600098) q[3];
rz(-1.8683627843856812) q[3];
rx(3.357543468475342) q[4];
ry(-3.086561918258667) q[4];
rz(1.300607681274414) q[4];
rx(-3.1415834426879883) q[5];
ry(-3.1415979862213135) q[5];
rz(-3.7090017795562744) q[5];
rx(-6.88179874420166) q[6];
ry(-3.4122419357299805) q[6];
rz(-1.4493516683578491) q[6];
rx(0.19584017992019653) q[7];
ry(6.044502258300781) q[7];
rz(-4.991343975067139) q[7];
rx(-0.07131747901439667) q[8];
ry(-0.26166969537734985) q[8];
rz(1.5117433071136475) q[8];
rx(-3.7052550315856934) q[9];
ry(0.3449822962284088) q[9];
rz(-6.719040870666504) q[9];
rx(-3.0864691734313965) q[10];
ry(0.19645747542381287) q[10];
rz(-2.9910686016082764) q[10];
rx(2.847423553466797) q[11];
ry(-2.992250442504883) q[11];
rz(-3.411959409713745) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(5.012564182281494) q[0];
ry(-1.5742672681808472) q[0];
rz(6.325847148895264) q[0];
rx(3.8893351554870605) q[1];
ry(1.7846107482910156) q[1];
rz(2.185960054397583) q[1];
rx(-3.141690254211426) q[2];
ry(0.0006523279589600861) q[2];
rz(0.16309626400470734) q[2];
rx(-3.1416468620300293) q[3];
ry(0.00019527976110111922) q[3];
rz(3.7705941200256348) q[3];
rx(4.835385322570801) q[4];
ry(0.2615053057670593) q[4];
rz(3.544034719467163) q[4];
rx(-3.1415905952453613) q[5];
ry(3.141676664352417) q[5];
rz(0.31317198276519775) q[5];
rx(3.2661044597625732) q[6];
ry(-9.242938995361328) q[6];
rz(-2.3580162525177) q[6];
rx(-2.959512948989868) q[7];
ry(-3.1643552780151367) q[7];
rz(-1.6007616519927979) q[7];
rx(-0.3426717519760132) q[8];
ry(-1.549845814704895) q[8];
rz(4.481343746185303) q[8];
rx(2.3144054412841797) q[9];
ry(-6.273892879486084) q[9];
rz(-0.4594617187976837) q[9];
rx(2.4743642807006836) q[10];
ry(6.426905632019043) q[10];
rz(-6.240937232971191) q[10];
rx(3.4649314880371094) q[11];
ry(-3.20212459564209) q[11];
rz(-0.9624658226966858) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-0.10165050625801086) q[0];
ry(2.5086793899536133) q[0];
rz(1.3604263067245483) q[0];
rx(1.2936981916427612) q[1];
ry(-4.849448204040527) q[1];
rz(-1.5219557285308838) q[1];
rx(3.6802265644073486) q[2];
ry(-5.449418544769287) q[2];
rz(-1.4019707441329956) q[2];
rx(1.2222851514816284) q[3];
ry(0.8210026621818542) q[3];
rz(1.2645355463027954) q[3];
rx(6.10876989364624) q[4];
ry(0.21027906239032745) q[4];
rz(2.4283714294433594) q[4];
rx(-1.7584919987712055e-05) q[5];
ry(3.141585350036621) q[5];
rz(-1.3287532329559326) q[5];
rx(4.018321514129639) q[6];
ry(-6.995658874511719) q[6];
rz(1.6325141191482544) q[6];
rx(-1.5180485248565674) q[7];
ry(1.9305223226547241) q[7];
rz(-0.5083580017089844) q[7];
rx(3.2464678287506104) q[8];
ry(-0.2567577660083771) q[8];
rz(0.9571563005447388) q[8];
rx(6.184723854064941) q[9];
ry(-0.014768274500966072) q[9];
rz(2.9249563217163086) q[9];
rx(1.841404914855957) q[10];
ry(5.7029643058776855) q[10];
rz(4.754037380218506) q[10];
rx(1.6864564418792725) q[11];
ry(-3.8281161785125732) q[11];
rz(-1.4272645711898804) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(-8.56159496307373) q[0];
ry(-8.107271194458008) q[0];
rz(-6.90699577331543) q[0];
rx(-3.9320802688598633) q[1];
ry(-0.6164146661758423) q[1];
rz(6.281150817871094) q[1];
rx(7.904719352722168) q[2];
ry(2.0782155990600586) q[2];
rz(-4.286394119262695) q[2];
rx(-3.494723081588745) q[3];
ry(-7.840328216552734) q[3];
rz(-1.3273463249206543) q[3];
rx(5.941002368927002) q[4];
ry(-4.78720235824585) q[4];
rz(3.8450403213500977) q[4];
rx(-3.1415655612945557) q[5];
ry(-6.283271789550781) q[5];
rz(2.1960017681121826) q[5];
rx(-0.42551034688949585) q[6];
ry(-4.142405033111572) q[6];
rz(0.007782769855111837) q[6];
rx(-4.271470546722412) q[7];
ry(-8.615327835083008) q[7];
rz(3.569230079650879) q[7];
rx(0.13973528146743774) q[8];
ry(-1.3963801860809326) q[8];
rz(1.6885229349136353) q[8];
rx(-9.127237319946289) q[9];
ry(4.690426349639893) q[9];
rz(3.4775853157043457) q[9];
rx(-1.4865463972091675) q[10];
ry(6.5192766189575195) q[10];
rz(-3.6961703300476074) q[10];
rx(2.4718377590179443) q[11];
ry(-4.599557876586914) q[11];
rz(4.978298664093018) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(3.7962424755096436) q[0];
ry(3.8101727962493896) q[0];
rz(-6.341302871704102) q[0];
rx(-2.684535503387451) q[1];
ry(-0.15877866744995117) q[1];
rz(3.4214999675750732) q[1];
rx(-1.3648728132247925) q[2];
ry(10.245101928710938) q[2];
rz(8.314128875732422) q[2];
rx(2.129204750061035) q[3];
ry(7.15659761428833) q[3];
rz(-2.9285521507263184) q[3];
rx(2.4747941493988037) q[4];
ry(-5.151149749755859) q[4];
rz(-3.028251886367798) q[4];
rx(-4.556756496429443) q[5];
ry(-2.9241724014282227) q[5];
rz(-1.7681827545166016) q[5];
rx(-1.4768972396850586) q[6];
ry(6.082446098327637) q[6];
rz(-1.4470775127410889) q[6];
rx(3.661771059036255) q[7];
ry(0.9424577951431274) q[7];
rz(-1.2473838329315186) q[7];
rx(-5.357022762298584) q[8];
ry(4.031896114349365) q[8];
rz(2.1472158432006836) q[8];
rx(6.3776655197143555) q[9];
ry(-0.34247493743896484) q[9];
rz(2.5123722553253174) q[9];
rx(-8.94507122039795) q[10];
ry(6.571791172027588) q[10];
rz(0.20735958218574524) q[10];
rx(3.0939395427703857) q[11];
ry(3.016275644302368) q[11];
rz(6.06771993637085) q[11];
rx(2.343907356262207) q[0];
ry(-0.15092693269252777) q[0];
rz(-6.682507514953613) q[0];
rx(-0.4443947374820709) q[1];
ry(-1.6748887300491333) q[1];
rz(3.5308806896209717) q[1];
rx(-1.7742068767547607) q[2];
ry(-5.33795166015625) q[2];
rz(-1.0707783699035645) q[2];
rx(-1.264102578163147) q[3];
ry(3.385226011276245) q[3];
rz(1.7597784996032715) q[3];
rx(1.3830240964889526) q[4];
ry(-3.795456886291504) q[4];
rz(-0.11900512874126434) q[4];
rx(-2.3254687786102295) q[5];
ry(1.7934744358062744) q[5];
rz(0.42293837666511536) q[5];
rx(5.719723701477051) q[6];
ry(11.009895324707031) q[6];
rz(-8.199804306030273) q[6];
rx(4.072402000427246) q[7];
ry(2.203411817550659) q[7];
rz(11.862260818481445) q[7];
rx(-3.5065462589263916) q[8];
ry(5.113124370574951) q[8];
rz(12.997200012207031) q[8];
rx(4.811214923858643) q[9];
ry(-4.7276787757873535) q[9];
rz(2.5034780502319336) q[9];
rx(-5.719166278839111) q[10];
ry(4.021453380584717) q[10];
rz(7.888942241668701) q[10];
rx(3.121115207672119) q[11];
ry(3.0095481872558594) q[11];
rz(2.1926963329315186) q[11];
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
rx(6.2845869064331055) q[0];
ry(0.0018275848124176264) q[0];
rz(1.1765637397766113) q[0];
rx(3.141631841659546) q[1];
ry(-3.141773223876953) q[1];
rz(-6.549707412719727) q[1];
rx(-1.5909303328953683e-05) q[2];
ry(7.34428476789617e-06) q[2];
rz(-3.084710121154785) q[2];
rx(-3.1413028240203857) q[3];
ry(-3.141637086868286) q[3];
rz(-7.861980438232422) q[3];
rx(-6.28298807144165) q[4];
ry(3.14127516746521) q[4];
rz(7.296831130981445) q[4];
rx(-3.141591787338257) q[5];
ry(-3.141592264175415) q[5];
rz(3.5115067958831787) q[5];
rx(3.0796725749969482) q[6];
ry(-3.0523905754089355) q[6];
rz(-9.336686134338379) q[6];
rx(9.418105125427246) q[7];
ry(0.018539760261774063) q[7];
rz(3.0753767490386963) q[7];
rx(7.958365440368652) q[8];
ry(-8.50644302368164) q[8];
rz(7.3219428062438965) q[8];
rx(3.1628952026367188) q[9];
ry(3.1310672760009766) q[9];
rz(7.114484786987305) q[9];
rx(-0.008545180782675743) q[10];
ry(-3.1368627548217773) q[10];
rz(4.158175945281982) q[10];
rx(-3.1416072845458984) q[11];
ry(-3.141500234603882) q[11];
rz(-1.929311752319336) q[11];
