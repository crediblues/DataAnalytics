create database titanic;
use titanic;
-- drop database if exists titanic;

-- How many passengers were travelling in each Passenger Class? 
select Pclass, count(Pclass) as 'No. of Passengers' from data 
group by pclass;

-- What was the proportion of males and females in each class?
select sex, count(sex) from data
group by sex;

-- What was the average fare in each class?
select pclass, avg(fare) from data
group by pclass;

-- Were there differences in fare by the gender of the passengers?
select sex, avg(fare) from data
group by sex;

-- Average age by class
select Pclass, avg(age) from data
group by Pclass;

-- What proportion of the total passengers survived?
select NumSurvived, TotalPassengers, NumSurvived/TotalPassengers from
(select count(*) as TotalPassengers from data) as a,
(select count(*) as NumSurvived from data where Survived = 1) as b;

-- Do the different passenger ticket classes differ in chances of survival? 
select Pclass, count(*) from data where Survived = 1
group by Pclass;
