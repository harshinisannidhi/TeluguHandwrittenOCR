create or replace database BBR;
use BBR;


create table users(
    id INT PRIMARY KEY AUTO_INCREMENT, 
    name VARCHAR(50), 
    email VARCHAR(50), 
    password VARCHAR(50)
    );