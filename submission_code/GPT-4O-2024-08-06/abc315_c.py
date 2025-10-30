# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    index = 1
    
    from collections import defaultdict
    flavor_map = defaultdict(list)
    
    for _ in range(N):
        F = int(data[index])
        S = int(data[index + 1])
        flavor_map[F].append(S)
        index += 2
    
    # To find the maximum satisfaction
    max_satisfaction = 0
    
    # To track the two largest deliciousness values across all different flavors
    top_two_different = [0, 0]
    
    for flavor, deliciousness_list in flavor_map.items():
        # Sort the deliciousness list in descending order
        deliciousness_list.sort(reverse=True)
        
        # Update the top two deliciousness values for different flavors
        for s in deliciousness_list[:2]:
            if s > top_two_different[0]:
                top_two_different[1] = top_two_different[0]
                top_two_different[0] = s
            elif s > top_two_different[1]:
                top_two_different[1] = s
        
        # Calculate satisfaction for the same flavor
        if len(deliciousness_list) >= 2:
            s1, s2 = deliciousness_list[0], deliciousness_list[1]
            satisfaction_same_flavor = s1 + s2 // 2
            max_satisfaction = max(max_satisfaction, satisfaction_same_flavor)
    
    # Calculate satisfaction for different flavors
    if top_two_different[1] > 0:
        satisfaction_different_flavors = top_two_different[0] + top_two_different[1]
        max_satisfaction = max(max_satisfaction, satisfaction_different_flavors)
    
    print(max_satisfaction)