from collections import deque

class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        if k >= 26:
            return 1
        
        dp_no = [0] * (n + 1)
        dp_yes = [0] * (n + 1)
        
        left_no = 0
        count_no = [0] * 26
        distinct_no = 0
        
        left_change = 0
        count_change = [0] * 26
        distinct_change = 0
        
        dq_no = deque()
        dq_yes = deque()
        dq_change = deque()
        
        max_partitions = 0
        
        for i in range(1, n + 1):
            c = ord(s[i-1]) - ord('a')
            
            # Update no-change window (k distinct)
            count_no[c] += 1
            if count_no[c] == 1:
                distinct_no += 1
            while distinct_no > k:
                left_char = ord(s[left_no]) - ord('a')
                count_no[left_char] -= 1
                if count_no[left_char] == 0:
                    distinct_no -= 1
                left_no += 1
            
            # Update dq_no for dp_no
            while dq_no and dq_no[0] < left_no:
                dq_no.popleft()
            while dq_no and dp_no[dq_no[-1]] <= dp_no[i-1]:
                dq_no.pop()
            dq_no.append(i-1)
            current_no = dp_no[dq_no[0]] + 1 if dq_no else 1
            dp_no[i] = current_no
            
            # Update change window (k+1 distinct)
            count_change[c] += 1
            if count_change[c] == 1:
                distinct_change += 1
            while distinct_change > k + 1:
                left_char_change = ord(s[left_change]) - ord('a')
                count_change[left_char_change] -= 1
                if count_change[left_char_change] == 0:
                    distinct_change -= 1
                left_change += 1
            
            # Update dq_change for dp_no in change window
            while dq_change and dq_change[0] < left_change:
                dq_change.popleft()
            while dq_change and dp_no[dq_change[-1]] <= dp_no[i-1]:
                dq_change.pop()
            dq_change.append(i-1)
            current_change = dp_no[dq_change[0]] + 1 if dq_change else 1
            
            # Update dq_yes for dp_yes in no-change window
            while dq_yes and dq_yes[0] < left_no:
                dq_yes.popleft()
            while dq_yes and dp_yes[dq_yes[-1]] <= dp_yes[i-1]:
                dq_yes.pop()
            dq_yes.append(i-1)
            
            case_a = dp_yes[dq_yes[0]] + 1 if dq_yes else 0
            case_b = current_change
            if case_a == 0 and case_b == 0:
                dp_yes[i] = 1
            else:
                dp_yes[i] = max(case_a, case_b)
            
            current_max = max(dp_no[i], dp_yes[i])
            if current_max > max_partitions:
                max_partitions = current_max
        
        return max_partitions