s1, s2 = input().split()

if s1 == 'sick' and s2 == 'fine':
    # Takahashi (ate 1, 2) got sick, Aoki (ate 1, 3) did not.
    # If 1 caused trouble, Aoki would also be sick.
    # If 3 caused trouble, Aoki would be sick, which he isn't.
    # Therefore, 2 must have caused trouble.
    print(2)
elif s1 == 'fine' and s2 == 'sick':
    # Takahashi (ate 1, 2) did not get sick, Aoki (ate 1, 3) got sick.
    # If 1 caused trouble, Takahashi would also be sick.
    # If 2 caused trouble, Takahashi would be sick, which he isn't.
    # Therefore, 3 must have caused trouble.
    print(3)
elif s1 == 'sick' and s2 == 'sick':
    # Takahashi (ate 1, 2) got sick, Aoki (ate 1, 3) got sick.
    # If 2 caused trouble, Aoki would not be sick.
    # If 3 caused trouble, Takahashi would not be sick.
    # The only oyster common to both that could make them both sick is 1.
    print(1)
elif s1 == 'fine' and s2 == 'fine':
    # Takahashi (ate 1, 2) did not get sick, Aoki (ate 1, 3) did not get sick.
    # This means 1, 2, and 3 are all fine.
    # Since exactly one oyster type causes trouble, it must be oyster 4.
    print(4)