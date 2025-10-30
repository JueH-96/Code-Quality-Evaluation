import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    P = list(map(int, data[1:1+n]))
    Q = list(map(int, data[1+n:1+2*n]))
    
    person_for_bib = [0] * (n + 1)
    for i in range(n):
        bib = Q[i]
        person_for_bib[bib] = i + 1
        
    res = []
    for bib in range(1, n + 1):
        j = person_for_bib[bib]
        target_id = P[j - 1]
        s_i = Q[target_id - 1]
        res.append(str(s_i))
        
    print(" ".join(res))

if __name__ == "__main__":
    main()