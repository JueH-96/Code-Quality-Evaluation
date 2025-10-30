def main():
    M = int(input().strip())
    S1 = input().strip()
    S2 = input().strip()
    S3 = input().strip()
    
    ans = 10**18
    
    for digit in '0123456789':
        list1 = [i for i, char in enumerate(S1) if char == digit]
        list2 = [i for i, char in enumerate(S2) if char == digit]
        list3 = [i for i, char in enumerate(S3) if char == digit]
        
        if not list1 or not list2 or not list3:
            continue
            
        for r1 in list1:
            for r2 in list2:
                for r3 in list3:
                    freq = {}
                    for r in [r1, r2, r3]:
                        freq[r] = freq.get(r, 0) + 1
                    candidate = 0
                    for r_val, count_val in freq.items():
                        t_val = r_val + (count_val - 1) * M
                        if t_val > candidate:
                            candidate = t_val
                    if candidate < ans:
                        ans = candidate
                        
    if ans == 10**18:
        print(-1)
    else:
        print(ans)

if __name__ == "__main__":
    main()