def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    cuboids = []
    for _ in range(n):
        x1 = int(input[idx])
        y1 = int(input[idx+1])
        z1 = int(input[idx+2])
        x2 = int(input[idx+3])
        y2 = int(input[idx+4])
        z2 = int(input[idx+5])
        idx += 6
        cuboids.append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'z1': z1, 'z2': z2, 'count': 0})
    
    # Process x-axis
    for a in range(101):
        left_group = []
        right_group = []
        for c in cuboids:
            if c['x1'] == a:
                left_group.append(c)
            if c['x2'] == a:
                right_group.append(c)
        # Sort left and right groups by y1
        left_group.sort(key=lambda c: c['y1'])
        right_group.sort(key=lambda c: c['y1'])
        # Process right group against left group
        for c in right_group:
            y1_i = c['y1']
            y2_i = c['y2']
            z1_i = c['z1']
            z2_i = c['z2']
            low = 0
            high = len(left_group) - 1
            found = -1
            while low <= high:
                mid = (low + high) // 2
                if left_group[mid]['y1'] < y2_i:
                    found = mid
                    low = mid + 1
                else:
                    high = mid - 1
            if found != -1:
                j = left_group[found]
                if j['y2'] > y1_i:
                    if j['z1'] < z2_i and j['z2'] > z1_i:
                        c['count'] += 1
        # Process left group against right group
        for c in left_group:
            y1_j = c['y1']
            y2_j = c['y2']
            z1_j = c['z1']
            z2_j = c['z2']
            low = 0
            high = len(right_group) - 1
            found = -1
            while low <= high:
                mid = (low + high) // 2
                if right_group[mid]['y1'] < y2_j:
                    found = mid
                    low = mid + 1
                else:
                    high = mid - 1
            if found != -1:
                i = right_group[found]
                if i['y2'] > y1_j:
                    if i['z1'] < z2_j and i['z2'] > z1_j:
                        c['count'] += 1
    
    # Process y-axis
    for a in range(101):
        left_group = []
        right_group = []
        for c in cuboids:
            if c['y1'] == a:
                left_group.append(c)
            if c['y2'] == a:
                right_group.append(c)
        # Sort left and right groups by x1
        left_group.sort(key=lambda c: c['x1'])
        right_group.sort(key=lambda c: c['x1'])
        # Process right group against left group
        for c in right_group:
            x1_i = c['x1']
            x2_i = c['x2']
            z1_i = c['z1']
            z2_i = c['z2']
            low = 0
            high = len(left_group) - 1
            found = -1
            while low <= high:
                mid = (low + high) // 2
                if left_group[mid]['x1'] < x2_i:
                    found = mid
                    low = mid + 1
                else:
                    high = mid - 1
            if found != -1:
                j = left_group[found]
                if j['x2'] > x1_i:
                    if j['z1'] < z2_i and j['z2'] > z1_i:
                        c['count'] += 1
        # Process left group against right group
        for c in left_group:
            x1_j = c['x1']
            x2_j = c['x2']
            z1_j = c['z1']
            z2_j = c['z2']
            low = 0
            high = len(right_group) - 1
            found = -1
            while low <= high:
                mid = (low + high) // 2
                if right_group[mid]['x1'] < x2_j:
                    found = mid
                    low = mid + 1
                else:
                    high = mid - 1
            if found != -1:
                i = right_group[found]
                if i['x2'] > x1_j:
                    if i['z1'] < z2_j and i['z2'] > z1_j:
                        c['count'] += 1
    
    # Process z-axis
    for a in range(101):
        left_group = []
        right_group = []
        for c in cuboids:
            if c['z1'] == a:
                left_group.append(c)
            if c['z2'] == a:
                right_group.append(c)
        # Sort left and right groups by x1
        left_group.sort(key=lambda c: c['x1'])
        right_group.sort(key=lambda c: c['x1'])
        # Process right group against left group
        for c in right_group:
            x1_i = c['x1']
            x2_i = c['x2']
            y1_i = c['y1']
            y2_i = c['y2']
            low = 0
            high = len(left_group) - 1
            found = -1
            while low <= high:
                mid = (low + high) // 2
                if left_group[mid]['x1'] < x2_i:
                    found = mid
                    low = mid + 1
                else:
                    high = mid - 1
            if found != -1:
                j = left_group[found]
                if j['x2'] > x1_i:
                    if j['y1'] < y2_i and j['y2'] > y1_i:
                        c['count'] += 1
        # Process left group against right group
        for c in left_group:
            x1_j = c['x1']
            x2_j = c['x2']
            y1_j = c['y1']
            y2_j = c['y2']
            low = 0
            high = len(right_group) - 1
            found = -1
            while low <= high:
                mid = (low + high) // 2
                if right_group[mid]['x1'] < x2_j:
                    found = mid
                    low = mid + 1
                else:
                    high = mid - 1
            if found != -1:
                i = right_group[found]
                if i['x2'] > x1_j:
                    if i['y1'] < y2_j and i['y2'] > y1_j:
                        c['count'] += 1
    
    for c in cuboids:
        print(c['count'])

if __name__ == '__main__':
    main()