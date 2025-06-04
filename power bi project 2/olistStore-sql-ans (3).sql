create database olistStore;
use olistStore;
show tables;

-- INSERTING TABLES
select * from olist_customers_dataset;
select * from olist_geolocation_dataset;
select * from olist_order_items_dataset;
select * from olist_order_payments_dataset;
select * from olist_order_reviews_dataset;
select * from olist_orders_dataset;
select * from olist_products_dataset;
select * from olist_sellers_dataset;
select * from product_category_name_translation;

-- 1). WEEKDAY VS WEEKEND PAYMENT STATISTICS:
SELECT order_purchase_timestamp,
       CASE WHEN DAYOFWEEK(order_purchase_timestamp as date) IN (1, 7) THEN 'weekend'
            ELSE 'weekday' END AS day_type
FROM olist_orders_dataset
LIMIT 0, 1000;
-- DATETIME IS CONVERTED TO DATE FORMAT
SELECT DATE(order_purchase_timestamp) AS purchase_date, 
       CASE WHEN DAYOFWEEK(order_purchase_timestamp) IN (1, 7) 
            THEN 'weekend'
            ELSE 'weekday' 
       END AS day_type 
FROM olist_orders_dataset ;

-- 2).Number of Orders with review score 5 and payment type as credit card
SELECT COUNT(*) AS num_people
FROM olist_order_reviews_dataset 
INNER JOIN olist_order_payments_dataset 
ON olist_order_reviews_dataset.order_id = olist_order_payments_dataset.order_id 
WHERE olist_order_reviews_dataset.review_score = 5 
AND olist_order_payments_dataset.payment_type = "credit_card" ;

-- 3) Average number of days taken for order_delivered_customer_date for pet_shop

select avg(datediff(order_delivered_customer_date, order_approved_at)) as avg_days 
from olist_orders_dataset 
inner join olist_order_items_dataset
on olist_orders_dataset.order_id=olist_order_items_dataset.order_id
inner join olist_products_dataset 
on olist_order_items_dataset.product_id=olist_products_dataset.product_id 
where product_category_name="pet_shop" ;

-- 4) Average price and payment values from customers of sao paulo city
select concat('R$',format(avg(price),2)) as avg_price,concat('R$',format(avg(payment_value),2)) as avg_payment_value from olist_order_payments_dataset 
inner join olist_order_items_dataset 
on olist_order_payments_dataset.order_id = olist_order_items_dataset.order_id
inner join olist_orders_dataset 
on olist_order_items_dataset.order_id = olist_orders_dataset.order_id
inner join olist_customers_dataset 
on olist_orders_dataset.customer_id = olist_customers_dataset.customer_id 
where customer_city="sao paulo";

-- 5) Relationship between shipping days (order_delivered_customer_date - order_purchase_timestamp) Vs review scores.
select order_purchase_timestamp as order_purchase,order_delivered_customer_date as order_deliver,review_score 
from olist_orders_dataset inner join olist_order_reviews_dataset 
on olist_orders_dataset.order_id = olist_order_reviews_dataset.order_id; 

-- using datediff()
select datediff(order_purchase_timestamp ,order_delivered_customer_date) as shipping_days,avg(review_score)
from olist_orders_dataset  join olist_order_reviews_dataset 
on olist_orders_dataset.order_id = olist_order_reviews_dataset.order_id group by shipping_days;



