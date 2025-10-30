def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    flavors = []
    deliciousness = []
    
    index = 1
    for _ in range(N):
        F = int(data[index])
        S = int(data[index + 1])
        flavors.append(F)
        deliciousness.append(S)
        index += 2
    
    from collections import defaultdict
    import heapq
    
    # Dictionary to store max two deliciousness values per flavor
    flavor_dict = defaultdict(list)
    
    for i in range(N):
        f = flavors[i]
        s = deliciousness[i]
        if len(flavor_dict[f]) < 2:
            heapq.heappush(flavor_dict[f], s)
        else:
            heapq.heappushpop(flavor_dict[f], s)
    
    # Now we have at most two largest values for each flavor
    # We need to find the two largest values overall and the best same-flavor pair
    max1 = -1
    max2 = -1
    best_same_flavor_satisfaction = -1
    
    for f, values in flavor_dict.items():
        if len(values) == 2:
            s1 = heapq.heappop(values)
            s2 = heapq.heappop(values)
            # Check for best same-flavor satisfaction
            current_same_flavor_satisfaction = s2 + s1 // 2
            if current_same_flavor_satisfaction > best_same_flavor_satisfaction:
                best_same_flavor_satisfaction = current_same_flavor_satisfaction
            
            # Push back to maintain state
            heapq.heappush(values, s1)
            heapq.heappush(values, s2)
        
        # Check for global max1 and max2
        for s in values:
            if s > max1:
                max2 = max1
                max1 = s
            elif s > max2:
                max2 = s
    
    # Best different-flavor satisfaction
    best_different_flavor_satisfaction = max1 + max2
    
    # The result is the maximum of the best same-flavor and different-flavor satisfactions
    result = max(best_same_flavor_satisfaction, best_different_flavor_satisfaction)
    
    print(result)

if __name__ == "__main__":
    main()