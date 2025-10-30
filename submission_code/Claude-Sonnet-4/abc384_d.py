n, s = map(int, input().split())
a = list(map(int, input().split()))

# Calculate prefix sums
prefix = [0]
for x in a:
    prefix.append(prefix[-1] + x)

period_sum = prefix[n]
found = False

# Check subsequences within first 2 periods
extended_a = a * 2
extended_prefix = [0]
for x in extended_a:
    extended_prefix.append(extended_prefix[-1] + x)

# Check all contiguous subsequences in the extended array
for i in range(len(extended_a)):
    for j in range(i + 1, len(extended_a) + 1):
        current_sum = extended_prefix[j] - extended_prefix[i]
        
        if current_sum == s:
            found = True
            break
            
        # If we can add complete periods to reach s
        if period_sum > 0 and current_sum < s:
            remaining = s - current_sum
            if remaining % period_sum == 0:
                found = True
                break
    
    if found:
        break

print("Yes" if found else "No")