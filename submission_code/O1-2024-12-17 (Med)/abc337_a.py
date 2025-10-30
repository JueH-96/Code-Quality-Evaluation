def main():
    import sys
    data = sys.stdin.read().strip().split()
    
    n = int(data[0])
    t_score = 0
    a_score = 0
    
    idx = 1
    for _ in range(n):
        x = int(data[idx])
        y = int(data[idx + 1])
        t_score += x
        a_score += y
        idx += 2
    
    if t_score > a_score:
        print("Takahashi")
    elif t_score < a_score:
        print("Aoki")
    else:
        print("Draw")

# Do not remove the line below
main()