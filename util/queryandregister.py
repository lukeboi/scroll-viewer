import os
import requests
import time
from register import brute_force_registration_3d

# Replace with your URLs and directory
# urls = [
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3891,3706,0&threshold=93&applySobel=true&cutoffPlane=0.9978561957524416,0.06544472934060441,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3771,3758,0&threshold=93&applySobel=true&cutoffPlane=0.9880602530117011,0.15406796038908663,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3704,3789,0&threshold=93&applySobel=true&cutoffPlane=0.9787050736023913,0.20527147611136298,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3610,3842,0&threshold=93&applySobel=true&cutoffPlane=0.9599832247796044,0.2800575086330515,0.0",
# ]
# urls = [ # first batch
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4090,3637,0&threshold=93&applySobel=true&cutoffPlane=-0.6656121658233237,-0.746297825742501,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3978,3673,0&threshold=93&applySobel=true&cutoffPlane=-0.6791244274479459,-0.7340231686013048,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3906,3708,0&threshold=93&applySobel=true&cutoffPlane=-0.6889681590735806,-0.7247916085212089,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3830,3729,0&threshold=93&applySobel=true&cutoffPlane=-0.6978431513916742,-0.7162506098117731,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3737,3781,0&threshold=93&applySobel=true&cutoffPlane=-0.7111264117702055,-0.7030641695342125,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3660,3811,0&threshold=93&applySobel=true&cutoffPlane=-0.7208861919334569,-0.6930534599002296,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3597,3850,0&threshold=93&applySobel=true&cutoffPlane=-0.7301018384430322,-0.6833383536009848,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3533,3908,0&threshold=93&applySobel=true&cutoffPlane=-0.7409179596435593,-0.6715955457547532,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3466,3979,0&threshold=93&applySobel=true&cutoffPlane=-0.7528624628464604,-0.6581778726429217,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3431,4019,0&threshold=93&applySobel=true&cutoffPlane=-0.7592175683497426,-0.650836910377019,0.0",
# ]
# urls = [
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4033,3664,50&threshold=88&applySobel=true&cutoffPlane=-0.998623542760535,0.052450165341949316,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3965,3688,50&threshold=88&applySobel=true&cutoffPlane=-0.9999841973513763,0.005621836668173585,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3888,3720,50&threshold=88&applySobel=true&cutoffPlane=-0.9987719576213065,-0.04954368445425604,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3804,3742,50&threshold=88&applySobel=true&cutoffPlane=-0.9938127068019058,-0.11106891463892739,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3726,3778,50&threshold=88&applySobel=true&cutoffPlane=-0.9853146535673902,-0.17074845084326118,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3632,3820,50&threshold=88&applySobel=true&cutoffPlane=-0.969743524781969,-0.244125984162773,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3563,3866,50&threshold=88&applySobel=true&cutoffPlane=-0.9533973085828764,-0.3017177024752235,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3530,3896,50&threshold=88&applySobel=true&cutoffPlane=-0.9434338834950876,-0.33156071461103076,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3499,3946,50&threshold=88&applySobel=true&cutoffPlane=-0.9306640784886271,-0.36587480510540016,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3445,3979,50&threshold=88&applySobel=true&cutoffPlane=-0.9111367073090907,-0.4121042351079986,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3396,4041,50&threshold=88&applySobel=true&cutoffPlane=-0.885653261206043,-0.46434717713700013,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3665,3888,50&threshold=88&applySobel=true&cutoffPlane=-0.9726609570839145,-0.23222976244358387,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3601,3943,50&threshold=88&applySobel=true&cutoffPlane=-0.9565547809097279,-0.29155265582522527,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3537,3984,50&threshold=88&applySobel=true&cutoffPlane=-0.9370447756039935,-0.3492092331443454,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3472,4029,50&threshold=88&applySobel=true&cutoffPlane=-0.9125201793777871,-0.40903168853810234,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3623,3771,50&threshold=88&applySobel=true&cutoffPlane=-0.9703126405076401,-0.24185404621608342,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3568,3799,50&threshold=88&applySobel=true&cutoffPlane=-0.9587464927206024,-0.2842624890764591,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3515,3847,50&threshold=88&applySobel=true&cutoffPlane=-0.9439514605746574,-0.3300842923844925,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3475,3891,50&threshold=88&applySobel=true&cutoffPlane=-0.930045557476602,-0.3674442284456739,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3421,3945,50&threshold=88&applySobel=true&cutoffPlane=-0.9085969291028296,-0.41767406003354773,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3747,3659,50&threshold=88&applySobel=true&cutoffPlane=-0.9897026226798517,-0.1431388090652678,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4123,3700,50&threshold=88&applySobel=true&cutoffPlane=-0.9931506043228763,0.11684124756739721,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4042,3733,50&threshold=88&applySobel=true&cutoffPlane=-0.9981029785032064,0.061566584305350176,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3967,3762,50&threshold=88&applySobel=true&cutoffPlane=-0.9999725256006401,0.007412694778359082,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3883,3807,50&threshold=88&applySobel=true&cutoffPlane=-0.9983936844691444,-0.056657310314966784,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3811,3850,50&threshold=88&applySobel=true&cutoffPlane=-0.9933640093159819,-0.11501280361628338,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4176,3761,50&threshold=88&applySobel=true&cutoffPlane=-0.9870961085972292,0.16012892428355052,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4096,3794,50&threshold=88&applySobel=true&cutoffPlane=-0.9944764518351209,0.10495992923696416,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3997,3821,50&threshold=88&applySobel=true&cutoffPlane=-0.9995196060492603,0.03099285600152745,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3933,3857,50&threshold=88&applySobel=true&cutoffPlane=-0.9998169043080801,-0.019135251757092443,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3867,3896,50&threshold=88&applySobel=true&cutoffPlane=-0.9972677544567091,-0.07387168551531179,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3707,3988,50&threshold=88&applySobel=true&cutoffPlane=-0.9761051842503016,-0.21729857173871361,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3639,4037,50&threshold=88&applySobel=true&cutoffPlane=-0.9588521353699886,-0.2839059395229575,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3570,4094,50&threshold=88&applySobel=true&cutoffPlane=-0.9346186967740141,-0.35565136248922663,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3526,4164,50&threshold=88&applySobel=true&cutoffPlane=-0.910168885925629,-0.41423737046879205,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3473,4223,50&threshold=88&applySobel=true&cutoffPlane=-0.8780466819895371,-0.4785749933366396,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3394,4159,50&threshold=88&applySobel=true&cutoffPlane=-0.8607464759953163,-0.5090338928417679,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3455,4111,50&threshold=88&applySobel=true&cutoffPlane=-0.8937110791016124,-0.4486429617090094,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3625,3820,100&threshold=88&applySobel=true&cutoffPlane=-0.9684877626657266,-0.24906114423316902,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3571,3854,100&threshold=88&applySobel=true&cutoffPlane=-0.9559432908587365,-0.29355140037507743,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3500,3919,100&threshold=88&applySobel=true&cutoffPlane=-0.9337286624970663,-0.3579815425848652,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3445,3977,100&threshold=88&applySobel=true&cutoffPlane=-0.9114094960862709,-0.4115005837708736,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3390,4039,100&threshold=88&applySobel=true&cutoffPlane=-0.8839683705310859,-0.46754670344321425,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3799,3911,100&threshold=88&applySobel=true&cutoffPlane=-0.9914430433540207,-0.1305400007082794,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3728,3978,100&threshold=88&applySobel=true&cutoffPlane=-0.9801793578597886,-0.19811215617819206,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3651,4038,100&threshold=88&applySobel=true&cutoffPlane=-0.9616591116706347,-0.274247612461523,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3602,4059,100&threshold=88&applySobel=true&cutoffPlane=-0.9475059227411964,-0.3197382153736927,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3568,4097,100&threshold=88&applySobel=true&cutoffPlane=-0.9336537646354239,-0.35817683870136086,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3518,4162,100&threshold=88&applySobel=true&cutoffPlane=-0.9075951550571317,-0.41984644159123374,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3470,4225,100&threshold=88&applySobel=true&cutoffPlane=-0.8763411775137924,-0.4816909181142403,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4086,4369,100&threshold=88&applySobel=true&cutoffPlane=-0.9852214984868823,0.1712851392248084,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4014,4362,100&threshold=88&applySobel=true&cutoffPlane=-0.9971168006032155,0.07588205291639957,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4000,3590,100&threshold=88&applySobel=true&cutoffPlane=-0.9996006182158161,0.028259583552452398,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3891,3617,100&threshold=88&applySobel=true&cutoffPlane=-0.9990256352193895,-0.04413366260005335,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3784,3642,100&threshold=88&applySobel=true&cutoffPlane=-0.9931367530548554,-0.11695892326650101,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3704,3674,100&threshold=88&applySobel=true&cutoffPlane=-0.984852476490473,-0.1733943469395196,0.0",
#     "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3615,3708,100&threshold=88&applySobel=true&cutoffPlane=-0.9715514356982555,-0.2368286464781207,0.0",
# ]
urls = [
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2283,3363,0&threshold=120&applySobel=true&cutoffPlane=-0.9824244517014221,-0.1866606458232702,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2478,4726,0&threshold=120&applySobel=true&cutoffPlane=0.9944122245726512,0.10556669749722772,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2387,4753,0&threshold=120&applySobel=true&cutoffPlane=0.9999198942602952,-0.012657213851396142,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2357,4742,0&threshold=120&applySobel=true&cutoffPlane=0.9986842986797899,-0.05128032342386598,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2626,4687,0&threshold=120&applySobel=true&cutoffPlane=0.9534432425640457,0.3015725173303404,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2606,4711,0&threshold=120&applySobel=true&cutoffPlane=0.9631109740028582,0.2691045368537398,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2662,4631,0&threshold=120&applySobel=true&cutoffPlane=0.9295285684251767,0.3687501057375327,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2676,4584,0&threshold=120&applySobel=true&cutoffPlane=0.9121686961106911,0.4098149214410351,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2647,4580,0&threshold=120&applySobel=true&cutoffPlane=0.9268099288828958,0.37553076534963364,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2677,4524,0&threshold=120&applySobel=true&cutoffPlane=0.8947459459922616,0.4465755167162803,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2716,4365,0&threshold=120&applySobel=true&cutoffPlane=0.7833340839833467,0.6216009273400188,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2684,4324,0&threshold=120&applySobel=true&cutoffPlane=0.7827690870859682,0.6223122659104512,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2643,4313,0&threshold=120&applySobel=true&cutoffPlane=0.818132254308594,0.5750300987426118,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2622,4303,0&threshold=120&applySobel=true&cutoffPlane=0.8339322057265399,0.5518669008484455,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2590,4351,0&threshold=120&applySobel=true&cutoffPlane=0.8953483253743311,0.44536656391042756,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2492,4407,0&threshold=120&applySobel=true&cutoffPlane=0.9778668283739254,0.2092282628277543,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2440,4252,0&threshold=120&applySobel=true&cutoffPlane=0.9891113879166765,0.14716882242358856,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2506,4231,0&threshold=120&applySobel=true&cutoffPlane=0.9263158466350108,0.3767478629970753,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2555,4181,0&threshold=120&applySobel=true&cutoffPlane=0.8096986867714758,0.5868458372013449,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2588,4110,0&threshold=120&applySobel=true&cutoffPlane=0.6099112533474766,0.7924697237371975,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2541,4095,0&threshold=120&applySobel=true&cutoffPlane=0.6757246285173463,0.7371541402007414,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2493,4148,0&threshold=120&applySobel=true&cutoffPlane=0.887609253901052,0.46059723445676215,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2355,4147,0&threshold=120&applySobel=true&cutoffPlane=0.9749242329362375,-0.22253705317022812,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2408,4143,0&threshold=120&applySobel=true&cutoffPlane=0.9981379299495438,0.06099731794136101,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2421,4159,0&threshold=120&applySobel=true&cutoffPlane=0.9925863886954068,0.1215411904524988,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2317,4591,0&threshold=120&applySobel=true&cutoffPlane=0.9919835173301202,-0.12636732704842296,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2422,4814,0&threshold=120&applySobel=true&cutoffPlane=0.9995687690640719,0.029364534931376963,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2564,4818,0&threshold=120&applySobel=true&cutoffPlane=0.9814537358449629,0.1916991507439869,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2568,4948,0&threshold=120&applySobel=true&cutoffPlane=0.985263100376391,0.1710456752937694,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2514,4963,0&threshold=120&applySobel=true&cutoffPlane=0.9932249786413646,0.11620732250103967,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2466,4950,0&threshold=120&applySobel=true&cutoffPlane=0.997565299424451,0.06973860755854824,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2585,4980,0&threshold=120&applySobel=true&cutoffPlane=0.9833396619792325,0.18177763662939597,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2582,5024,0&threshold=120&applySobel=true&cutoffPlane=0.9851366992020528,0.17177218600601296,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2551,5029,0&threshold=120&applySobel=true&cutoffPlane=0.9897254373432723,0.14298097312463784,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2499,5005,0&threshold=120&applySobel=true&cutoffPlane=0.9952430628407204,0.09742302534525286,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2531,5130,0&threshold=120&applySobel=true&cutoffPlane=0.9934721657830983,0.11407478167518009,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2622,5137,0&threshold=120&applySobel=true&cutoffPlane=0.9821255894973987,0.1882267952614265,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2670,5124,0&threshold=120&applySobel=true&cutoffPlane=0.9734501467623149,0.22889913011723684,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2616,5185,0&threshold=120&applySobel=true&cutoffPlane=0.9843178606899082,0.1764039373904173,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2548,5175,0&threshold=120&applySobel=true&cutoffPlane=0.9923281796222719,0.12363164614105863,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2575,5232,0&threshold=120&applySobel=true&cutoffPlane=0.990305280351426,0.13890806926915195,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2606,5255,0&threshold=120&applySobel=true&cutoffPlane=0.9871673987372204,0.15968884391337387,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2544,5272,0&threshold=120&applySobel=true&cutoffPlane=0.9937534357129264,0.11159797941161205,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2469,3890,0&threshold=120&applySobel=true&cutoffPlane=-0.7119664484531625,0.7022134834058589,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2500,3949,0&threshold=120&applySobel=true&cutoffPlane=-0.1346838895922121,0.9908886162855606,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2511,4026,0&threshold=120&applySobel=true&cutoffPlane=0.29784011941241095,0.9546157673474709,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2466,4077,0&threshold=120&applySobel=true&cutoffPlane=0.7474093186836598,0.6643638388299198,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2411,4146,0&threshold=120&applySobel=true&cutoffPlane=0.9878635102232196,0.15532445129295905,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2324,4146,0&threshold=120&applySobel=true&cutoffPlane=0.9316743891413858,-0.36329441589160955,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2276,4087,0&threshold=120&applySobel=true&cutoffPlane=0.6726727939963125,-0.7399400733959437,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2258,3997,0&threshold=120&applySobel=true&cutoffPlane=0.07788766729290965,-0.9969621413492434,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2262,3931,0&threshold=120&applySobel=true&cutoffPlane=-0.4115867224264983,-0.9113705996586747,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2294,3870,0&threshold=120&applySobel=true&cutoffPlane=-0.7860851426451102,-0.6181182318235054,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2349,3816,0&threshold=120&applySobel=true&cutoffPlane=-0.9773822453765274,-0.21148036888264043,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2449,3763,0&threshold=120&applySobel=true&cutoffPlane=-0.962650940153899,0.2707455769182841,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2486,3760,0&threshold=120&applySobel=true&cutoffPlane=-0.9151366082629749,0.40314388029205944,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2549,3778,0&threshold=120&applySobel=true&cutoffPlane=-0.7885388984102495,0.6149848824922041,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2602,3815,0&threshold=120&applySobel=true&cutoffPlane=-0.6229267817697037,0.782280144548,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2629,3849,0&threshold=120&applySobel=true&cutoffPlane=-0.4938249551411479,0.8695613340528908,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2781,3852,0&threshold=120&applySobel=true&cutoffPlane=-0.3234054803718959,0.9462604796066584,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2723,3857,0&threshold=120&applySobel=true&cutoffPlane=-0.3599064960866811,0.9329883783170119,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2699,3816,0&threshold=120&applySobel=true&cutoffPlane=-0.479441346154219,0.8775739260015822,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2270,3683,0&threshold=120&applySobel=true&cutoffPlane=-0.9342928493282386,-0.3565064819805121,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2261,3677,0&threshold=120&applySobel=true&cutoffPlane=-0.9274414720324122,-0.3739683354969404,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2324,3632,0&threshold=120&applySobel=true&cutoffPlane=-0.9850893068591293,-0.17204376626835496,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2233,3602,0&threshold=120&applySobel=true&cutoffPlane=-0.9293066983280764,-0.3693088956992096,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2298,3552,0&threshold=120&applySobel=true&cutoffPlane=-0.9801449732436269,-0.1982822014837682,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2221,3553,0&threshold=120&applySobel=true&cutoffPlane=-0.9347263560205139,-0.3553683150769235,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2337,3508,0&threshold=120&applySobel=true&cutoffPlane=-0.9948084283649208,-0.10176537158639064,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2368,3475,0&threshold=120&applySobel=true&cutoffPlane=-0.999382591304283,-0.035134544225541194,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2327,3437,0&threshold=120&applySobel=true&cutoffPlane=-0.9942954674648262,-0.10666078650986317,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2302,3344,0&threshold=120&applySobel=true&cutoffPlane=-0.9915745941385717,-0.129536960976112,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2237,3413,0&threshold=120&applySobel=true&cutoffPlane=-0.9679209933081849,-0.25125475261832675,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2202,3463,0&threshold=120&applySobel=true&cutoffPlane=-0.9435210027097114,-0.33131271850875366,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2325,3383,0&threshold=120&applySobel=true&cutoffPlane=-0.9949388526067533,-0.1004822351142582,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2435,3346,0&threshold=120&applySobel=true&cutoffPlane=-0.9970909725387914,0.07622068276817595,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2479,3417,0&threshold=120&applySobel=true&cutoffPlane=-0.9869497337019628,0.16102864076189918,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2502,3471,0&threshold=120&applySobel=true&cutoffPlane=-0.9756502324316491,0.2193322227947118,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2916,2852,0&threshold=120&applySobel=true&cutoffPlane=-0.9060811398238874,0.4231039683759122,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2911,2921,0&threshold=120&applySobel=true&cutoffPlane=-0.8971039694211641,0.4418194971351887,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2916,2993,0&threshold=120&applySobel=true&cutoffPlane=-0.8824020839604764,0.47049608098496226,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2934,3055,0&threshold=120&applySobel=true&cutoffPlane=-0.8620293078750729,0.5068584342441416,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2998,3114,0&threshold=120&applySobel=true&cutoffPlane=-0.8188351394109185,0.57402875752518,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2906,3209,0&threshold=120&applySobel=true&cutoffPlane=-0.8313920927742761,0.5556862316743233,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2957,3272,0&threshold=120&applySobel=true&cutoffPlane=-0.7814016321331195,0.6240284362909249,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2496,2760,0&threshold=120&applySobel=true&cutoffPlane=-0.9960055406024937,0.08929145025776226,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=3161,2906,0&threshold=120&applySobel=true&cutoffPlane=-0.8127162565846272,0.5826596659140482,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=3170,2963,0&threshold=120&applySobel=true&cutoffPlane=-0.7940055545690287,0.6079105027169126,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=3194,3006,0&threshold=120&applySobel=true&cutoffPlane=-0.7718848081397544,0.6357624107817752,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=3214,3057,0&threshold=120&applySobel=true&cutoffPlane=-0.7468773757390574,0.6649617925934834,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=3015,3353,0&threshold=120&applySobel=true&cutoffPlane=-0.7099005325704472,0.7043019479287244,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=2998,3278,0&threshold=120&applySobel=true&cutoffPlane=-0.7569910331087261,0.6534252641220597,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=4214,3895,0&threshold=120&applySobel=true&cutoffPlane=-0.05026460944977528,0.9987359355890132,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=4231,3928,0&threshold=120&applySobel=true&cutoffPlane=-0.03196198159112204,0.9994890853494943,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=4528,4090,0&threshold=120&applySobel=true&cutoffPlane=0.04803040384568024,0.9988458741499716,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=4591,3930,0&threshold=120&applySobel=true&cutoffPlane=-0.025841707345649185,0.9996660473185343,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=4601,3909,0&threshold=120&applySobel=true&cutoffPlane=-0.03519263327261031,0.999380547420921,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=4425,4731,0&threshold=120&applySobel=true&cutoffPlane=0.3427786437508784,0.9394162024301627,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=4405,4778,0&threshold=120&applySobel=true&cutoffPlane=0.36478185834295956,0.9310930113709677,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=4385,4816,0&threshold=120&applySobel=true&cutoffPlane=0.3830727245795629,0.9237181862901642,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=3918,5928,0&threshold=120&applySobel=true&cutoffPlane=0.7849552222858222,0.6195524989911795,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=3905,6001,0&threshold=120&applySobel=true&cutoffPlane=0.7983797204256469,0.6021543174411905,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=3935,5998,0&threshold=120&applySobel=true&cutoffPlane=0.7922284104395833,0.6102246682103005,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=3958,5974,0&threshold=120&applySobel=true&cutoffPlane=0.7842458658977671,0.6204501767444841,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=852,5278,0&threshold=120&applySobel=true&cutoffPlane=0.6439053308894275,-0.7651051724123794,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=846,5216,0&threshold=120&applySobel=true&cutoffPlane=0.6237661798929339,-0.7816109984012354,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=830,5140,0&threshold=120&applySobel=true&cutoffPlane=0.5953633782565777,-0.8034565625041066,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=805,5067,0&threshold=120&applySobel=true&cutoffPlane=0.564065886430139,-0.8257297837463424,0.0',
'http://localhost:5001/volume?filename=scroll1&size=200,200,100&origin=824,5728,0&threshold=120&applySobel=true&cutoffPlane=0.744334913871073,-0.667806510894093,0.0',
]
positions = [[3891, 3706, 0], [3771, 3758, 0], [3704, 3789, 0]]
directory = "./segmentations"

# Dictionary to store paths of newest directories for each URL
segmentations = {}

i = 0

# Loop through each URL
for url in urls:
    print(i, len(urls))
    i += 1
    print(f"=====Getting {url}")
    # Send a GET request to the current URL
    response = requests.get(url)

    # If the response is 200, find the newest directory in a folder
    if response.status_code == 200:
        print(f"Response 200 received for {url}.")

        # List all directories in the given directory
        all_subdirs = [
            os.path.join(directory, d)
            for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ]

        if all_subdirs:  # If list is not empty
            # Find the newest directory
            newest_directory = max(all_subdirs, key=os.path.getmtime)

            # Add the path of the newest directory to the dictionary
            segmentations[url] = newest_directory

            print(f"Newest segmentation for {url}: {newest_directory}")
        else:
            print("No directories found.")
    else:
        print(f"Unexpected response received for {url}: {response.status_code}")

# Print out all the newest directories
for url, directory in segmentations.items():
    print(f"{directory}")


# registration_src_folder = segmentations[urls[0]]
# segmentation_candidates = urls[1:]

# for segmentation in segmentation_candidates:
#     segmentation_folder = segmentations[segmentation]

#     # register
#     # Provide your folder paths and overlap percentage
#     volume1_folder_path = segmentation_folder + "/volume" # super simple first segments
#     volume2_folder_path = registration_src_folder + "/volume"
#     mask_file_path = segmentation_folder + "/mask.png"
#     mask2_file_path = registration_src_folder + "/mask.png"
#     output_folder_path = f"./registrations/{int(time.time())}"

#     brute_force_registration_3d(volume1_folder_path, volume2_folder_path, mask_file_path, mask2_file_path, output_folder_path, overlap_percent=0.8, min_distance=50)

#     registration_src_folder = output_folder_path

# print()
# print("FINAL REGISTRATION:", registration_src_folder)
