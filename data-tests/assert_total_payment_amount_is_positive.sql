
select
    order_id,
    sum(amount) as total_amount
from {{ ref('fct_orders') }}
group by 1
having not(total_amount >= 0)