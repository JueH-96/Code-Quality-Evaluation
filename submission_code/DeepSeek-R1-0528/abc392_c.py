import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    P = list(map(int, data[1:1+n]))
    Q = list(map(int, data[1+n:1+2*n]))
    
    b2p = [0] * (n+1)
    for idx in range(n):
        bib_val = Q[idx]
        b2p[bib_val] = idx + 1
        
    res = []
    for bib_i in range(1, n+1):
        person_x = b2p[bib_i]
        target_person = P[person_x - 1]
        bib_target = Q[target_person - 1]
        res.append(str(bib_target))
        
    print(" ".join(res))

if __name__ == '__main__':
    main()