def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    
    ans = []
    prev_h = -1
    current_ans = 0
    for h in H:
        if not ans:
            current_ans = h + 1
        else:
            # Compute the current answer based on the previous answer and the current H
            current_ans = current_ans + max(h - prev_h, 0) + 1
        ans.append(current_ans)
        prev_h = max(prev_h, h)
    print(' '.join(map(str, ans)))

if __name__ == "__main__":
    main()