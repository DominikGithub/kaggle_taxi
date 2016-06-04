# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact



#### Data file formats ####
weather: [21 x 3 x 144]=[days x types x timeslots]
traffic: [30 x 66 x 144 x 4]=[day, distr, dtime_slt, type]
demand/supply: [30 x 66 x 144]=[day, distr, dtime_slt]
gap: [7 x 66 x 144]
start/destination: [30 x 66=[days, distr]
pois: [66 x 30 x 2]
pois_simple: [66 x 30]