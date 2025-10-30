# YOUR CODE HERE
t = int(input())
for _ in range(t):
    s = input().strip()
    if s == 'abc':
        print("YES")
        continue
    # Try swapping each pair
    swapped = False
    for i in range(2):
        for j in range(i+1, 3):
            # Create a list to manipulate
            s_list = list(s)
            s_list[i], s_list[j] = s_list[j], s_list[i]
            if ''.join(s_list) == 'abc':
                print("YES")
                swapped = True
                break
        if swapped:
            break
    if not swapped:
        print("NO")