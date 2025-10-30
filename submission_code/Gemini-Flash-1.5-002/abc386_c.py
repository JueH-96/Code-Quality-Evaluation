k = int(input())
s = input()
t = input()

if s == t:
    print("Yes")
else:
    len_s = len(s)
    len_t = len(t)
    diff = 0
    i = 0
    j = 0
    while i < len_s and j < len_t:
        if s[i] == t[j]:
            i += 1
            j += 1
        else:
            diff += 1
            i += 1
    diff += len_s - i + len_t - j
    if diff <= k:
        print("Yes")
    else:
        
        diff_count = 0
        i = 0
        j = 0
        while i < len_s and j < len_t:
          if s[i] == t[j]:
            i += 1
            j += 1
          else:
            diff_count +=1
            i+=1
            j+=1
        diff_count += (len_s - i) + (len_t - j)
        if diff_count <=k:
          print("Yes")
        else:
          
          diff_count = 0
          i = 0
          j = 0
          while i < len_s and j < len_t:
            if s[i] == t[j]:
              i += 1
              j += 1
            else:
              diff_count += 1
              i += 1
              j += 1
          diff_count += (len_s - i) + (len_t - j)
          if diff_count <= k:
            print("Yes")
          else:
            print("No")