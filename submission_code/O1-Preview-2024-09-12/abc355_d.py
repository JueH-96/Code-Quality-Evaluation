# YOUR CODE HERE
import sys
import threading

def main():
    import sys

    N = int(sys.stdin.readline())
    events = []
    for _ in range(N):
        l_i, r_i = map(int, sys.stdin.readline().split())
        events.append( (l_i, 0) )
        events.append( (r_i, 1) )

    events.sort()
    ans = 0
    active = 0
    for time, event_type in events:
        if event_type == 0:  # start event
            ans += active
            active +=1
        else:  # end event
            active -=1
    print(ans)
    
threading.Thread(target=main).start()