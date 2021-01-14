import sys

if len(sys.argv) == 2:
    arg = sys.argv[1]
else:
    print('Exactly one argument is permitted.')
    exit()

if arg in ['-l', '--load_data']:
    from conversion import Main
    call = lambda: Main()
elif arg in ['-f', '--flush_data']:
    from conversion import Main
    call = lambda: Main(True)
elif arg in ['-t', '--train']:
    from trainModel import Main
    call = lambda: Main()
elif arg in ['-p', '--predict']:
    from useModel import Main
    call = lambda: Main()
elif arg in ['-h', '--help']:
    print('Usage: Voice2Voice.py [-l | --load_data] [-f | --flush_data] [-t | --train] [-p | --predict]')
    exit()
else:
    print('Invalid argument usage.\nUse -h or --help for help.')
    exit()

call()
