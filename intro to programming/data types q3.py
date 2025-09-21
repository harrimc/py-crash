def get_expected_cost(beds, baths, hasbasement) :
    if hasbasement == True :
        cost = (80000 + (beds * 30000)+ (baths * 10000) + 40000)
        return cost
    else :
        cost1 = (80000 + (beds * 30000)+ (baths * 10000))
        return cost1

print('To estimate your house cost we need to know 3 things:')

print('1. How many bedrooms do you have?')
bed = int(input('> '))

print('2. How many bathrooms do you have?')
bath = int(input('> '))

print('3. Do you have a basement?')
base_in = input('> ')
if base_in == "yes" :
    base = True
else :
    base = False

tot = get_expected_cost(bed, bath, base)

print('Your house price is ', tot)