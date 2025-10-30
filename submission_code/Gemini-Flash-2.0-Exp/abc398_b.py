from collections import Counter

cards = list(map(int, input().split()))
counts = Counter(cards)

nums = list(counts.keys())
n = len(nums)

if n > 2:
    print("No")
else:
    if n == 1:
        print("No")
    else:
        c1 = counts[nums[0]]
        c2 = counts[nums[1]]
        
        if (c1 == 3 and c2 == 2) or (c1 == 2 and c2 == 3):
            print("Yes")
        elif (c1 >= 3 and c2 >= 2) or (c1 >= 2 and c2 >= 3):
            
            if c1 >= 3 and c2 >= 2:
                if c1 + c2 >= 5:
                    print("Yes")
                else:
                    print("No")
            elif c1 >= 2 and c2 >= 3:
                if c1 + c2 >= 5:
                    print("Yes")
                else:
                    print("No")
            else:
                print("No")
        else:
            print("No")