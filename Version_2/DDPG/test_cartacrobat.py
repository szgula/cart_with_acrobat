import gym
from getch import getch
from threading import Thread
from time import sleep


### BACKGROUND CODE
def KeyCheck():
    global Break_KeyCheck
    global key
    Break_KeyCheck = False

    while Break_KeyCheck:
        base = getch()
        if base == '\xe0':
            sub = getch()

            if sub == 'H':
                key = 'UP_KEY'
            elif sub == 'M':
                key = 'RIGHT_KEY'
            elif sub == 'P':
                key = 'DOWN_KEY'
            elif sub == 'K':
                key = 'LEFT_KEY'
            else:
                key = None



def main():
    Thread(target=KeyCheck).start()

    env_name = "CartAcrobat-v0"
    env = gym.make(env_name)

    ok = True
    counter = 0
    state = env.reset()
    while ok:
        counter += 1
        if counter > 1000:
            break
        print(key)



if __name__ == "__main__":
    main()