def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    seq = []
    for x in a:
        seq.append(x)
        while len(seq) >= 2 and seq[-1] == seq[-2]:
            seq.pop()
            seq[-1] += 1
    
    print(len(seq))

solve()