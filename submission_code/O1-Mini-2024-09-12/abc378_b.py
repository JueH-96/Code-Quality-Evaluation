# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    q = []
    r = []
    for _ in range(N):
        qi = int(data[idx]); ri = int(data[idx+1]); idx +=2
        q.append(qi)
        r.append(ri)
    Q = int(data[idx]); idx +=1
    for _ in range(Q):
        tj = int(data[idx])-1; dj = int(data[idx+1]); idx +=2
        qi = q[tj]
        ri = r[tj]
        dj_mod_q = dj % qi
        delta = (ri - dj_mod_q + qi) % qi
        answer = dj + delta
        print(answer)

if __name__ == "__main__":
    main()