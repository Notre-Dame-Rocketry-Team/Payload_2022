import RPi.GPIO as GPIO
import time,threading

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

baudrate = OneBitDelay = timeout = Tx = Rx = timeout_exit = False

def begin(tx=2,rx=3,Baudrate=9600,Timeout=float('inf')):
    global Tx,Rx,baudrate,OneBitDelay,timeout
    Tx = tx
    Rx = rx
    baudtate = Baudrate
    timeout = Timeout
    GPIO.setup(Tx, GPIO.OUT, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(Rx, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    OneBitDelay = 1/baudrate



def setBaudrate(BaudRate):
    global Baudrate,OneBitDelay
    Baudrate = BaudRate
    OneBitDelay = 1/Baudrate

def mytimer():
    global timeout_exit
    timeout_exit = True

def read(byte = 0):
    global Tx,Rx,OneBitDelay,timeout,timeout_exit
    data_array = ""
    if timeout != float('inf'):
        timer = threading.Timer(timeout, mytimer) 
        timer.start()
    while GPIO.input(Rx):

        if timeout_exit:
            timeout_exit = False
            return None
    data = readValue = ""
    if byte == 0:
        while True:
            time.sleep(OneBitDelay/baudrate)      ## I think synchronization problem arries due to this delay
            for count in range(0,8):
                readValue = readValue + str(GPIO.input(Rx))
                time.sleep(OneBitDelay)
            if readValue != "11111111":
                print("Received binary ",readValue)
                data = data + chr(int(readValue, 2))
                readValue = ""
            else:
                return data
    else:
        for r in range(0,int(byte/8)):
            for count in range(0,8):
                readValue = readValue + str(GPIO.input(Rx))
                time.sleep(OneBitDelay)
            data = data + chr(int(readValue, 2))
            readValue = ""
        return(data)
def write(data):
    global OneBitDelay
    if type(data) == int:
        data = str(data)
    data = getbinarystring(data)
    dataTemp =""

    for r in range(0,data.count(" ")+1):
        dataTemp = dataTemp + data.split()[r].zfill(8)
    for sendBit in range(0,len(dataTemp)):
        GPIO.output(Tx, int(dataTemp[sendBit]))
        time.sleep(OneBitDelay)
    GPIO.output(Tx, True)
    time.sleep(.005)   ## I think synchronization problem aeries due to this delay

def getbinarystring(data):
    return ' '.join(format(ord(x), 'b') for x in data)
