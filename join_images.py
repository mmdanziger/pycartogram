from glob import glob
from sys import argv
import datetime
from os.path import join
from subprocess import call

input_dir = argv[1]
output_dir = argv[2]

week1_start = datetime.datetime(2020,1,6)
week2_start = datetime.timedelta(days=77) + week1_start
week_jump = datetime.timedelta(days=14)
current_time = week1_start
frame_no=1
while current_time < datetime.datetime(2020,3,17):
    f1 = glob(join(input_dir,f"*{current_time.isoformat()}*"))[0]
    other_time = current_time + week_jump
    try:
        f2 = glob(join(input_dir,f"*{other_time.isoformat()}*"))[0]
    except:
        break
    
    of = join(output_dir,"%05d.png"%frame_no)
    shlist = ["convert", "+append", f1, f2, of]
    call(shlist)
    current_time += datetime.timedelta(hours=1)
    frame_no+=1
    
