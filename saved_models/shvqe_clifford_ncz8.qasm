OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
rx(2.104783535003662) q[0];
ry(-2.4822869300842285) q[0];
rz(1.0971225500106812) q[0];
rx(0.357391893863678) q[1];
ry(-3.031712055206299) q[1];
rz(4.343355178833008) q[1];
rx(-4.039181232452393) q[2];
ry(-1.414128065109253) q[2];
rz(1.2320448160171509) q[2];
rx(-3.1416022777557373) q[3];
ry(3.141591787338257) q[3];
rz(-1.3348268270492554) q[3];
rx(0.5632051229476929) q[4];
ry(-2.7529349327087402) q[4];
rz(1.8854901790618896) q[4];
rx(-2.603931188583374) q[5];
ry(-1.996125340461731) q[5];
rz(-6.060609340667725) q[5];
rx(-2.5560009479522705) q[6];
ry(2.402289867401123) q[6];
rz(-1.7736576795578003) q[6];
rx(-3.809339761734009) q[7];
ry(-2.805389404296875) q[7];
rz(-3.3331286907196045) q[7];
rx(0.006093520205467939) q[8];
ry(0.04675181582570076) q[8];
rz(-1.45225989818573) q[8];
rx(3.1411828994750977) q[9];
ry(3.1434109210968018) q[9];
rz(0.8592581748962402) q[9];
rx(-5.9221696853637695) q[10];
ry(1.012097716331482) q[10];
rz(0.4873805344104767) q[10];
rx(1.435402274131775) q[11];
ry(2.5452351570129395) q[11];
rz(2.4730801582336426) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(-3.37915301322937) q[0];
ry(-2.0976357460021973) q[0];
rz(-4.986648082733154) q[0];
rx(-3.9781928062438965) q[1];
ry(0.6361936926841736) q[1];
rz(1.7392956018447876) q[1];
rx(0.16795390844345093) q[2];
ry(-1.6527305841445923) q[2];
rz(1.9418164491653442) q[2];
rx(-3.068087339401245) q[3];
ry(0.01437794417142868) q[3];
rz(-4.364617824554443) q[3];
rx(0.17621397972106934) q[4];
ry(-2.4895517826080322) q[4];
rz(4.572665214538574) q[4];
rx(-1.9194250106811523) q[5];
ry(-3.683222532272339) q[5];
rz(-1.0038753747940063) q[5];
rx(0.18656031787395477) q[6];
ry(-0.973345160484314) q[6];
rz(-4.328334331512451) q[6];
rx(-1.862447738647461) q[7];
ry(-3.5272791385650635) q[7];
rz(-2.0559630393981934) q[7];
rx(3.0658092498779297) q[8];
ry(0.1036488264799118) q[8];
rz(3.751311779022217) q[8];
rx(5.816656112670898) q[9];
ry(0.5433976650238037) q[9];
rz(1.454233169555664) q[9];
rx(-3.313525676727295) q[10];
ry(4.178248882293701) q[10];
rz(2.156806707382202) q[10];
rx(1.6046302318572998) q[11];
ry(2.345092296600342) q[11];
rz(2.0967845916748047) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(2.3465168476104736) q[0];
ry(-2.452815294265747) q[0];
rz(1.6089760065078735) q[0];
rx(-4.3455328941345215) q[1];
ry(-2.1916255950927734) q[1];
rz(3.5809013843536377) q[1];
rx(-3.987672805786133) q[2];
ry(-3.7270920276641846) q[2];
rz(-3.766420602798462) q[2];
rx(-3.1642069816589355) q[3];
ry(-0.07928616553544998) q[3];
rz(1.9427772760391235) q[3];
rx(3.1410069465637207) q[4];
ry(-3.1411521434783936) q[4];
rz(0.5739731192588806) q[4];
rx(3.797086000442505) q[5];
ry(-1.7751986980438232) q[5];
rz(2.831871509552002) q[5];
rx(6.79305362701416) q[6];
ry(-1.1754624843597412) q[6];
rz(1.5789778232574463) q[6];
rx(-2.9495255947113037) q[7];
ry(-3.145124912261963) q[7];
rz(2.371828556060791) q[7];
rx(2.0555243492126465) q[8];
ry(-0.3901642858982086) q[8];
rz(5.778346061706543) q[8];
rx(5.624664306640625) q[9];
ry(-3.6795337200164795) q[9];
rz(3.986844539642334) q[9];
rx(1.192365050315857) q[10];
ry(1.3676354885101318) q[10];
rz(1.3384612798690796) q[10];
rx(3.7825942039489746) q[11];
ry(1.7357827425003052) q[11];
rz(3.166408061981201) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(3.3583145141601562) q[0];
ry(-2.9267125129699707) q[0];
rz(1.2018102407455444) q[0];
rx(-2.946441173553467) q[1];
ry(3.0985653400421143) q[1];
rz(2.441655158996582) q[1];
rx(-2.2409253120422363) q[2];
ry(-0.3385154902935028) q[2];
rz(-0.20951752364635468) q[2];
rx(-3.1549668312072754) q[3];
ry(0.015962693840265274) q[3];
rz(3.001004219055176) q[3];
rx(3.2129404544830322) q[4];
ry(-4.651771068572998) q[4];
rz(-0.06181105226278305) q[4];
rx(3.5806784629821777) q[5];
ry(-3.952543020248413) q[5];
rz(2.020235061645508) q[5];
rx(-3.9259984493255615) q[6];
ry(-3.9880783557891846) q[6];
rz(0.01618666760623455) q[6];
rx(-6.277706146240234) q[7];
ry(-3.134101390838623) q[7];
rz(3.7887983322143555) q[7];
rx(-2.9494247436523438) q[8];
ry(-3.462024211883545) q[8];
rz(-3.6159543991088867) q[8];
rx(3.309723377227783) q[9];
ry(-2.989969491958618) q[9];
rz(-0.3579433858394623) q[9];
rx(-0.01643209159374237) q[10];
ry(3.339242458343506) q[10];
rz(0.07784616947174072) q[10];
rx(4.618770599365234) q[11];
ry(2.3230979442596436) q[11];
rz(2.5544636249542236) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(-0.08073559403419495) q[0];
ry(-3.048104763031006) q[0];
rz(1.1823357343673706) q[0];
rx(-3.225109100341797) q[1];
ry(0.322544127702713) q[1];
rz(0.6119949221611023) q[1];
rx(-4.45621919631958) q[2];
ry(-3.1320178508758545) q[2];
rz(-2.3058478832244873) q[2];
rx(-2.8408215045928955) q[3];
ry(1.5497084856033325) q[3];
rz(3.8736867904663086) q[3];
rx(2.035189628601074) q[4];
ry(-1.7257239818572998) q[4];
rz(0.4006359875202179) q[4];
rx(2.994748830795288) q[5];
ry(-4.712313652038574) q[5];
rz(4.4468159675598145) q[5];
rx(2.621760368347168) q[6];
ry(0.5944790840148926) q[6];
rz(1.6707112789154053) q[6];
rx(-1.7714534997940063) q[7];
ry(4.666043758392334) q[7];
rz(0.8706744909286499) q[7];
rx(-0.5995674729347229) q[8];
ry(-7.045106410980225) q[8];
rz(0.6283295750617981) q[8];
rx(-0.014375779777765274) q[9];
ry(-0.12666989862918854) q[9];
rz(1.4248870611190796) q[9];
rx(3.492358922958374) q[10];
ry(3.0200512409210205) q[10];
rz(-0.09422186762094498) q[10];
rx(-2.202665090560913) q[11];
ry(1.0406907796859741) q[11];
rz(0.5683696269989014) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(1.1389975547790527) q[0];
ry(-0.39841192960739136) q[0];
rz(0.34759968519210815) q[0];
rx(-2.4424936771392822) q[1];
ry(-1.4596728086471558) q[1];
rz(3.345815658569336) q[1];
rx(-2.3152756690979004) q[2];
ry(-5.790472030639648) q[2];
rz(-3.323185920715332) q[2];
rx(-1.5522762537002563) q[3];
ry(4.38493013381958) q[3];
rz(1.143826961517334) q[3];
rx(0.047858234494924545) q[4];
ry(-4.181815147399902) q[4];
rz(-4.742035865783691) q[4];
rx(1.831209421157837) q[5];
ry(-4.835200786590576) q[5];
rz(0.400319904088974) q[5];
rx(3.1415934562683105) q[6];
ry(4.372081093606539e-05) q[6];
rz(0.31968727707862854) q[6];
rx(-4.284238338470459) q[7];
ry(-3.9899189472198486) q[7];
rz(-1.1426150798797607) q[7];
rx(0.6357417106628418) q[8];
ry(3.0503897666931152) q[8];
rz(-3.0563035011291504) q[8];
rx(3.2553811073303223) q[9];
ry(-1.5962507724761963) q[9];
rz(4.891567230224609) q[9];
rx(-5.737945079803467) q[10];
ry(1.111671805381775) q[10];
rz(1.968255639076233) q[10];
rx(-0.9718570113182068) q[11];
ry(2.1238229274749756) q[11];
rz(-0.7542162537574768) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(3.606785297393799) q[0];
ry(-3.459320545196533) q[0];
rz(4.608824729919434) q[0];
rx(-2.3238563537597656) q[1];
ry(-0.8217369914054871) q[1];
rz(-1.2214877605438232) q[1];
rx(-3.1421258449554443) q[2];
ry(-0.0011497727828100324) q[2];
rz(-2.051541328430176) q[2];
rx(-3.1337027549743652) q[3];
ry(3.144537925720215) q[3];
rz(3.1966469287872314) q[3];
rx(0.8886812329292297) q[4];
ry(-1.6801944971084595) q[4];
rz(-2.5519819259643555) q[4];
rx(3.141591787338257) q[5];
ry(-3.1416015625) q[5];
rz(-0.24247761070728302) q[5];
rx(-3.141582489013672) q[6];
ry(-3.1415884494781494) q[6];
rz(-3.400552988052368) q[6];
rx(-3.1843998432159424) q[7];
ry(-6.284234046936035) q[7];
rz(-1.0801013708114624) q[7];
rx(0.0006078784354031086) q[8];
ry(3.1393628120422363) q[8];
rz(-3.3058104515075684) q[8];
rx(-0.49034583568573) q[9];
ry(-2.0033390522003174) q[9];
rz(-3.7296769618988037) q[9];
rx(0.062484581023454666) q[10];
ry(-0.3102739751338959) q[10];
rz(-1.2906049489974976) q[10];
rx(-1.562099814414978) q[11];
ry(2.3238513469696045) q[11];
rz(-1.562461256980896) q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
rx(-0.4657857120037079) q[0];
ry(-1.4849416017532349) q[0];
rz(5.6926093101501465) q[0];
rx(-4.002612113952637) q[1];
ry(4.354414939880371) q[1];
rz(-1.9485859870910645) q[1];
rx(-1.5701627731323242) q[2];
ry(-0.8270772695541382) q[2];
rz(0.9491682052612305) q[2];
rx(-4.72454309463501) q[3];
ry(1.8809726238250732) q[3];
rz(3.3377633094787598) q[3];
rx(1.8610820770263672) q[4];
ry(-1.6157275438308716) q[4];
rz(-0.2835434377193451) q[4];
rx(1.4513072967529297) q[5];
ry(-1.60154390335083) q[5];
rz(3.9982898235321045) q[5];
rx(5.761971473693848) q[6];
ry(5.277762413024902) q[6];
rz(-2.7232089042663574) q[6];
rx(-4.699723243713379) q[7];
ry(-4.312054634094238) q[7];
rz(4.239745140075684) q[7];
rx(1.829038143157959) q[8];
ry(1.5143296718597412) q[8];
rz(-2.9520363807678223) q[8];
rx(1.503303050994873) q[9];
ry(-7.369978904724121) q[9];
rz(0.6167096495628357) q[9];
rx(-1.0596380233764648) q[10];
ry(5.72922420501709) q[10];
rz(-2.255425214767456) q[10];
rx(-3.586040735244751) q[11];
ry(1.5869542360305786) q[11];
rz(0.5820556879043579) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
rx(5.730539321899414) q[0];
ry(-6.236876010894775) q[0];
rz(-0.27791354060173035) q[0];
rx(-3.3077805042266846) q[1];
ry(2.3404505252838135) q[1];
rz(-1.2185686826705933) q[1];
rx(-3.853346109390259) q[2];
ry(3.0184969902038574) q[2];
rz(0.5410276651382446) q[2];
rx(0.32983893156051636) q[3];
ry(4.66325569152832) q[3];
rz(-0.2923421859741211) q[3];
rx(-1.5186803340911865) q[4];
ry(5.972124099731445) q[4];
rz(-3.151491403579712) q[4];
rx(-3.5358357429504395) q[5];
ry(-4.692819595336914) q[5];
rz(3.929725408554077) q[5];
rx(-4.001867771148682) q[6];
ry(-1.7432864904403687) q[6];
rz(-5.4902448654174805) q[6];
rx(-2.245779037475586) q[7];
ry(-2.2905380725860596) q[7];
rz(-3.5393149852752686) q[7];
rx(-5.210769176483154) q[8];
ry(-5.431649684906006) q[8];
rz(2.4365322589874268) q[8];
rx(3.565580368041992) q[9];
ry(-3.8698980808258057) q[9];
rz(3.208843231201172) q[9];
rx(1.3192981481552124) q[10];
ry(0.7528120875358582) q[10];
rz(-0.044318392872810364) q[10];
rx(-1.5744330883026123) q[11];
ry(3.1999666690826416) q[11];
rz(1.5681146383285522) q[11];
rx(5.114507675170898) q[0];
ry(-0.7102351188659668) q[0];
rz(5.226297378540039) q[0];
rx(-3.1727001667022705) q[1];
ry(0.3118766248226166) q[1];
rz(-7.569526195526123) q[1];
rx(-3.36118221282959) q[2];
ry(1.859156608581543) q[2];
rz(1.6378273963928223) q[2];
rx(-4.6103010177612305) q[3];
ry(4.87547492980957) q[3];
rz(-0.934757649898529) q[3];
rx(0.005762437358498573) q[4];
ry(1.266894817352295) q[4];
rz(-8.111047744750977) q[4];
rx(-3.530994415283203) q[5];
ry(-4.0422163009643555) q[5];
rz(-1.7935566902160645) q[5];
rx(-3.854369640350342) q[6];
ry(-2.887381076812744) q[6];
rz(-4.488856315612793) q[6];
rx(-1.7617007493972778) q[7];
ry(-1.5265690088272095) q[7];
rz(-0.1819991022348404) q[7];
rx(-2.9778409004211426) q[8];
ry(-4.130005836486816) q[8];
rz(-1.9642879962921143) q[8];
rx(1.8139854669570923) q[9];
ry(-3.847259521484375) q[9];
rz(4.528232097625732) q[9];
rx(2.7250471115112305) q[10];
ry(2.96588134765625) q[10];
rz(-2.197551965713501) q[10];
rx(4.174984455108643) q[11];
ry(1.5755869150161743) q[11];
rz(5.389918327331543) q[11];
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
