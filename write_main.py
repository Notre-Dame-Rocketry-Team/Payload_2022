import myuart,time
myuart.begin(tx=4,rx =17,Baudrate = 9600)
while True:
    myuart.write("hello")
    time.sleep(1)
