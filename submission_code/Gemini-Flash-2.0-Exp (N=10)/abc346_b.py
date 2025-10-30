def solve():
    w, b = map(int, input().split())
    s = "wbwbwwbwbwbw"
    
    for length in range(1, 1000):
        for start in range(len(s)):
            sub = ""
            for i in range(length):
                sub += s[(start + i) % len(s)]
            
            w_count = sub.count('w')
            b_count = sub.count('b')
            
            if w_count == w and b_count == b:
                print("Yes")
                return
    
    print("No")

solve()