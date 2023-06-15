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
urls = [
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4033,3664,50&threshold=88&applySobel=true&cutoffPlane=-0.998623542760535,0.052450165341949316,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3965,3688,50&threshold=88&applySobel=true&cutoffPlane=-0.9999841973513763,0.005621836668173585,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3888,3720,50&threshold=88&applySobel=true&cutoffPlane=-0.9987719576213065,-0.04954368445425604,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3804,3742,50&threshold=88&applySobel=true&cutoffPlane=-0.9938127068019058,-0.11106891463892739,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3726,3778,50&threshold=88&applySobel=true&cutoffPlane=-0.9853146535673902,-0.17074845084326118,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3632,3820,50&threshold=88&applySobel=true&cutoffPlane=-0.969743524781969,-0.244125984162773,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3563,3866,50&threshold=88&applySobel=true&cutoffPlane=-0.9533973085828764,-0.3017177024752235,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3530,3896,50&threshold=88&applySobel=true&cutoffPlane=-0.9434338834950876,-0.33156071461103076,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3499,3946,50&threshold=88&applySobel=true&cutoffPlane=-0.9306640784886271,-0.36587480510540016,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3445,3979,50&threshold=88&applySobel=true&cutoffPlane=-0.9111367073090907,-0.4121042351079986,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3396,4041,50&threshold=88&applySobel=true&cutoffPlane=-0.885653261206043,-0.46434717713700013,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3665,3888,50&threshold=88&applySobel=true&cutoffPlane=-0.9726609570839145,-0.23222976244358387,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3601,3943,50&threshold=88&applySobel=true&cutoffPlane=-0.9565547809097279,-0.29155265582522527,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3537,3984,50&threshold=88&applySobel=true&cutoffPlane=-0.9370447756039935,-0.3492092331443454,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3472,4029,50&threshold=88&applySobel=true&cutoffPlane=-0.9125201793777871,-0.40903168853810234,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3623,3771,50&threshold=88&applySobel=true&cutoffPlane=-0.9703126405076401,-0.24185404621608342,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3568,3799,50&threshold=88&applySobel=true&cutoffPlane=-0.9587464927206024,-0.2842624890764591,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3515,3847,50&threshold=88&applySobel=true&cutoffPlane=-0.9439514605746574,-0.3300842923844925,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3475,3891,50&threshold=88&applySobel=true&cutoffPlane=-0.930045557476602,-0.3674442284456739,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3421,3945,50&threshold=88&applySobel=true&cutoffPlane=-0.9085969291028296,-0.41767406003354773,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3747,3659,50&threshold=88&applySobel=true&cutoffPlane=-0.9897026226798517,-0.1431388090652678,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4123,3700,50&threshold=88&applySobel=true&cutoffPlane=-0.9931506043228763,0.11684124756739721,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4042,3733,50&threshold=88&applySobel=true&cutoffPlane=-0.9981029785032064,0.061566584305350176,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3967,3762,50&threshold=88&applySobel=true&cutoffPlane=-0.9999725256006401,0.007412694778359082,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3883,3807,50&threshold=88&applySobel=true&cutoffPlane=-0.9983936844691444,-0.056657310314966784,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3811,3850,50&threshold=88&applySobel=true&cutoffPlane=-0.9933640093159819,-0.11501280361628338,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4176,3761,50&threshold=88&applySobel=true&cutoffPlane=-0.9870961085972292,0.16012892428355052,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4096,3794,50&threshold=88&applySobel=true&cutoffPlane=-0.9944764518351209,0.10495992923696416,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3997,3821,50&threshold=88&applySobel=true&cutoffPlane=-0.9995196060492603,0.03099285600152745,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3933,3857,50&threshold=88&applySobel=true&cutoffPlane=-0.9998169043080801,-0.019135251757092443,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3867,3896,50&threshold=88&applySobel=true&cutoffPlane=-0.9972677544567091,-0.07387168551531179,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3707,3988,50&threshold=88&applySobel=true&cutoffPlane=-0.9761051842503016,-0.21729857173871361,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3639,4037,50&threshold=88&applySobel=true&cutoffPlane=-0.9588521353699886,-0.2839059395229575,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3570,4094,50&threshold=88&applySobel=true&cutoffPlane=-0.9346186967740141,-0.35565136248922663,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3526,4164,50&threshold=88&applySobel=true&cutoffPlane=-0.910168885925629,-0.41423737046879205,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3473,4223,50&threshold=88&applySobel=true&cutoffPlane=-0.8780466819895371,-0.4785749933366396,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3394,4159,50&threshold=88&applySobel=true&cutoffPlane=-0.8607464759953163,-0.5090338928417679,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3455,4111,50&threshold=88&applySobel=true&cutoffPlane=-0.8937110791016124,-0.4486429617090094,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3625,3820,100&threshold=88&applySobel=true&cutoffPlane=-0.9684877626657266,-0.24906114423316902,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3571,3854,100&threshold=88&applySobel=true&cutoffPlane=-0.9559432908587365,-0.29355140037507743,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3500,3919,100&threshold=88&applySobel=true&cutoffPlane=-0.9337286624970663,-0.3579815425848652,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3445,3977,100&threshold=88&applySobel=true&cutoffPlane=-0.9114094960862709,-0.4115005837708736,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3390,4039,100&threshold=88&applySobel=true&cutoffPlane=-0.8839683705310859,-0.46754670344321425,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3799,3911,100&threshold=88&applySobel=true&cutoffPlane=-0.9914430433540207,-0.1305400007082794,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3728,3978,100&threshold=88&applySobel=true&cutoffPlane=-0.9801793578597886,-0.19811215617819206,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3651,4038,100&threshold=88&applySobel=true&cutoffPlane=-0.9616591116706347,-0.274247612461523,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3602,4059,100&threshold=88&applySobel=true&cutoffPlane=-0.9475059227411964,-0.3197382153736927,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3568,4097,100&threshold=88&applySobel=true&cutoffPlane=-0.9336537646354239,-0.35817683870136086,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3518,4162,100&threshold=88&applySobel=true&cutoffPlane=-0.9075951550571317,-0.41984644159123374,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3470,4225,100&threshold=88&applySobel=true&cutoffPlane=-0.8763411775137924,-0.4816909181142403,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4086,4369,100&threshold=88&applySobel=true&cutoffPlane=-0.9852214984868823,0.1712851392248084,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4014,4362,100&threshold=88&applySobel=true&cutoffPlane=-0.9971168006032155,0.07588205291639957,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=4000,3590,100&threshold=88&applySobel=true&cutoffPlane=-0.9996006182158161,0.028259583552452398,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3891,3617,100&threshold=88&applySobel=true&cutoffPlane=-0.9990256352193895,-0.04413366260005335,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3784,3642,100&threshold=88&applySobel=true&cutoffPlane=-0.9931367530548554,-0.11695892326650101,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3704,3674,100&threshold=88&applySobel=true&cutoffPlane=-0.984852476490473,-0.1733943469395196,0.0",
    "http://localhost:5001/volume?filename=campfire&size=200,200,100&origin=3615,3708,100&threshold=88&applySobel=true&cutoffPlane=-0.9715514356982555,-0.2368286464781207,0.0",
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
