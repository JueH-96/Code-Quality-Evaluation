import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    x_list = list(map(int, input[idx:idx+Q]))
    
    current_S = set()
    event_dict = defaultdict(list)
    s = [0]  # s[0] is the initial size (0)
    
    for i in range(1, Q + 1):
        x = x_list[i - 1]
        if x in current_S:
            current_S.remove(x)
            event_dict[x].append(('remove', i))
        else:
            current_S.add(x)
            event_dict[x].append(('add', i))
        s_i = len(current_S)
        s.append(s_i)
    
    # Compute prefix sum array
    pre = [0]
    current_pre = 0
    for i in range(1, Q + 1):
        current_pre += s[i]
        pre.append(current_pre)
    
    # Initialize the result array
    A = [0] * (N + 1)  # A[0] is unused
    
    for x in event_dict:
        events = event_dict[x]
        i = 0
        while i < len(events):
            if events[i][0] == 'add':
                t_add = events[i][1]
                if i + 1 < len(events) and events[i + 1][0] == 'remove':
                    t_remove = events[i + 1][1]
                    sum_contribution = pre[t_remove - 1] - pre[t_add - 1]
                    A[x] += sum_contribution
                    i += 2
                else:
                    # x was added but not removed
                    sum_contribution = pre[Q] - pre[t_add - 1]
                    A[x] += sum_contribution
                    i += 1
            else:
                # This case should not occur as the first event for x is 'add'
                i += 1
    
    # Prepare the output
    output = []
    for j in range(1, N + 1):
        output.append(str(A[j]))
    print(' '.join(output))

if __name__ == '__main__':
    main()