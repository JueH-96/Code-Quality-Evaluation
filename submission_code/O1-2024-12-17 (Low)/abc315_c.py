def main():
    import sys
    data = sys.stdin.read().strip().split()
    n = int(data[0])
    
    # Dictionary to store the maximum deliciousness for each flavor
    max_for_flavor = {}
    # Dictionary to store up to the top 2 deliciousness values for each flavor
    top2_for_flavor = {}
    
    idx = 1
    for _ in range(n):
        f = int(data[idx]); s = int(data[idx+1])
        idx += 2
        
        # Track maximum deliciousness per flavor
        if f not in max_for_flavor:
            max_for_flavor[f] = s
        else:
            if s > max_for_flavor[f]:
                max_for_flavor[f] = s
        
        # Track top 2 deliciousness per flavor
        if f not in top2_for_flavor:
            top2_for_flavor[f] = [s]
        else:
            if len(top2_for_flavor[f]) < 2:
                top2_for_flavor[f].append(s)
            else:
                # If s is bigger than the smaller of the top 2, update
                mn = min(top2_for_flavor[f])
                if s > mn:
                    top2_for_flavor[f].remove(mn)
                    top2_for_flavor[f].append(s)
    
    # Calculate the maximum satisfaction if eating from different flavors
    # We'll sum the two largest distinct-flavor values
    max_del_list = list(max_for_flavor.values())
    max_del_list.sort(reverse=True)
    diff_flavor_ans = 0
    if len(max_del_list) >= 2:
        diff_flavor_ans = max_del_list[0] + max_del_list[1]
    
    # Calculate the maximum satisfaction if eating from the same flavor
    same_flavor_ans = 0
    for f, arr in top2_for_flavor.items():
        if len(arr) == 2:
            arr.sort(reverse=True)
            a, b = arr
            # Since b is even, b//2 is integer
            val = a + b // 2
            if val > same_flavor_ans:
                same_flavor_ans = val
    
    # Print the maximum of the two scenarios
    print(max(diff_flavor_ans, same_flavor_ans))

# Don't forget to call main()
if __name__ == "__main__":
    main()