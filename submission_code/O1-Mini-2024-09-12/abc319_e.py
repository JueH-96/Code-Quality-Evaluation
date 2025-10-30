# YOUR CODE HERE
import sys
import numpy as np

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    X = int(data[idx]); idx +=1
    Y = int(data[idx]); idx +=1
    P = []
    T = []
    for _ in range(N-1):
        p = int(data[idx]); idx +=1
        t = int(data[idx]); idx +=1
        P.append(p)
        T.append(t)
    Q = int(data[idx]); idx +=1
    queries = list(map(int, data[idx:idx+Q]))
    LCM = 840
    t_arr = np.arange(LCM, dtype=np.int64) + X
    for i in range(N-1):
        pi = P[i]
        ti = T[i]
        mod = t_arr % pi
        wait = (pi - mod) % pi
        t_arr += wait + ti
    t_arr += Y
    journey_time = t_arr - (np.arange(LCM, dtype=np.int64) + X)
    output = []
    jt = journey_time
    for q in queries:
        r = (q + X) % LCM
        arrival = q + X + jt[r]
        output.append(str(arrival))
    print('
'.join(output))

if __name__ == "__main__":
    main()