class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + int(s[i])
        
        from collections import defaultdict
        
        count3 = defaultdict(int)
        count6 = defaultdict(int)
        count9 = defaultdict(int)
        count3[0] = 1  # initial prefix_sum[0] mod3 is 0
        count6[0] = 1
        count9[0] = 1
        
        total = 0
        
        for i in range(n):
            d = int(s[i])
            if d == 0:
                continue
            
            if d in [1, 2, 5]:
                cnt = i + 1
                total += cnt
            elif d in [3, 6, 9]:
                current_mod = prefix_sum[i+1] % d
                # The count is the number of times current_mod has been seen in prefix_sum[0..i]
                # which is count_dict[d][current_mod] before updating
                if d == 3:
                    cnt = count3[current_mod]
                elif d == 6:
                    cnt = count6[current_mod]
                else:  # d ==9
                    cnt = count9[current_mod]
                total += cnt
                # Update the count_dict for future use
                if d ==3:
                    count3[current_mod] +=1
                elif d ==6:
                    count6[current_mod] +=1
                else:
                    count9[current_mod] +=1
            elif d ==4:
                if i ==0:
                    cnt =1
                else:
                    last_two = int(s[i-1])*10 + int(s[i])
                    if last_two %4 ==0:
                        cnt = i+1
                    else:
                        cnt =1
                total += cnt
            elif d ==8:
                if i <2:
                    if i ==0:
                        cnt =1
                    else:
                        last_two = int(s[0])*10 + int(s[1])
                        if last_two %8 ==0:
                            cnt =2
                        else:
                            cnt =1
                else:
                    last_three = int(s[i-2])*100 + int(s[i-1])*10 + int(s[i])
                    if last_three %8 ==0:
                        last_two = int(s[i-1])*10 + int(s[i])
                        valid_2 = (last_two %8 ==0)
                        cnt = 1 + (1 if valid_2 else 0) + (i-1)
                    else:
                        last_two = int(s[i-1])*10 + int(s[i])
                        valid_2 = (last_two %8 ==0)
                        cnt = 1 + (1 if valid_2 else 0)
                total += cnt
            elif d ==7:
                # Brute-force approach for d=7 (inefficient but for demonstration)
                current_mod = 0
                power =1
                cnt =0
                for j in range(i, -1, -1):
                    digit = int(s[j])
                    current_mod = (digit * power + current_mod) %7
                    if current_mod ==0:
                        cnt +=1
                    power = (power *10) %7
                total += cnt
        
        return total