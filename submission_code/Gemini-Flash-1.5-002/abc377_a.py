s = input()
sorted_s = sorted(s)
if sorted_s == ['A', 'B', 'C']:
    print('Yes')
elif sorted_s == ['A', 'C', 'B']:
    print('Yes')
elif sorted_s == ['B', 'A', 'C']:
    print('Yes')
elif sorted_s == ['B', 'C', 'A']:
    print('No')
elif sorted_s == ['C', 'A', 'B']:
    print('Yes')
elif sorted_s == ['C', 'B', 'A']:
    print('No')
else:
    print('No')