def main():
    t = int(input().strip())
    for _ in range(t):
        s = input().strip()
        
        # If no swap is needed
        if s == "abc":
            print("YES")
            continue
        
        # Check all possible swaps (at most one swap)
        s_list = list(s)
        possible = False
        for i in range(3):
            for j in range(i+1, 3):
                s_list[i], s_list[j] = s_list[j], s_list[i]
                if "".join(s_list) == "abc":
                    possible = True
                # swap back to restore original configuration
                s_list[i], s_list[j] = s_list[j], s_list[i]
        
        print("YES" if possible else "NO")

# Do not forget to call main()
main()