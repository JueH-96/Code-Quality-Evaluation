def largest_good_sequence(start, end):
    if start == 0:
        # For start = 0, the good sequence is S(0, 2^i) for the largest i s.t. 2^i <= end
        i = 0
        while (1 << (i + 1)) <= end:
            i += 1
        return 1 << i
    
    # Find the largest power of 2 that divides start (i.e., 2^i where i is the number of trailing zeros in start)
    power = start & -start
    
    # Ensure that start + power <= end
    while power > 1 and start + power > end:
        power >>= 1
    
    return start + power

def divide_into_good_sequences(L, R):
    result = []
    current = L
    
    while current < R:
        next_current = largest_good_sequence(current, R)
        result.append((current, next_current))
        current = next_current
    
    return result

L, R = map(int, input().split())
divisions = divide_into_good_sequences(L, R)
print(len(divisions))
for start, end in divisions:
    print(start, end)