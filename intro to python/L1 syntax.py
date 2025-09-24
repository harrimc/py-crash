alice_candies = 121
bob_candies = 77
carol_candies = 109 

candies_each = (alice_candies + bob_candies + carol_candies)/3

to_smash = 0 

if candies_each != int(candies_each) :
    ## using the round function ensuers that the terminating decimal will return either 1 or 2 
    to_smash = round((candies_each- int(candies_each))*3) 
## makes use of that int will always round down
candies_each = int(candies_each)

print('They each get ', candies_each, " candies")
print('the smashed', to_smash, ' candies')