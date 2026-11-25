# How the demo documents were synthesised

Both demo corpora are **LLM-synthesised**, which the assignment permits
("Public documents or LLM-synthesized ones are both fine; commit the synthesis
prompts"). Nothing here is a real Meridian Maritime Services document; the
company does not exist. Synthetic documents were chosen over public ones for
three reasons: the content can be pinned to exact numbers so expected facts in
the eval set are checkable, deliberate coverage gaps can be planted to exercise
the groundability guard, and there is no licensing question about committing
them.

The plain-text sources are in `demos/<run>/source/`. The PDFs the factory
actually ingests are generated from them by `scripts/make_demo_pdfs.py`, so the
corpora exercise real PDF text extraction. The CSV is committed directly.

## Synthesis prompt - demo 1, HR policy (3 PDFs)

> Write three internal HR policy documents for a fictional ship-management
> company, Meridian Maritime Services, covering its **shore-based** (office)
> staff. One document on leave, one on pay and benefits, one on workplace
> conduct and hybrid working.
>
> Constraints:
> - Write them as real policy documents: a document number, a revision, an
>   effective date, numbered sections, flat declarative sentences.
> - Every section must contain at least one hard, checkable number - a number of
>   days, a percentage, a rate, a threshold, a deadline. These are what the
>   generated eval set will grade against, so they must be unambiguous and must
>   not contradict each other across the three documents.
> - Put a few facts where answering correctly needs two documents: notice
>   periods appear in the pay document while leave-on-termination appears in the
>   leave document; probation gates both the equipment allowance and the cycle
>   scheme.
> - **Deliberately omit anything about seagoing personnel**: no shipboard
>   rotations, no sea-service days, no certificate revalidation. Scope the leave
>   policy explicitly to shore staff so the omission is visible in the text.
> - No real company names, no real people, no real monetary jurisdictions.

The omission in the last constraint is the point of this demo. The brief in
`demos/hr_policy/brief.txt` asks for shipboard rotation entitlements and
sea-service days for certificate revalidation, and the documents cannot support
either. A factory that generates an agent which answers those questions anyway
has failed core requirement 2.

## Synthesis prompt - demo 2, ship spares procurement (1 CSV + 2 PDFs)

> Write a ship-spares dataset and two supporting policy documents for the same
> fictional company, for its technical purchasing team.
>
> 1. A CSV catalogue of about 24 spare parts with the columns: part_no,
>    description, category, vessel_class, unit_price_usd, lead_time_days,
>    supplier, stock_qty, min_order_qty. Spread the rows across propulsion,
>    auxiliary, pumps, safety, electrical, deck, HVAC and water treatment. Vary
>    the prices across three orders of magnitude and the lead times from 7 to 60
>    days. Include at least one part with zero stock and at least one with a lead
>    time long enough to matter. Use exactly four supplier names.
> 2. A procurement policy PDF with value-banded approval thresholds, quotation
>    count rules banded by value, an emergency procurement route with its own
>    limit and ratification deadline, reorder rules, and freight/delivery notice
>    rules.
> 3. A supplier commercial terms PDF covering those same four suppliers:
>    payment terms, incoterms, warranty, volume discounts, returns, and 2025
>    on-time delivery performance.
>
> Constraints:
> - The three documents must interlock: answering "can I order this today
>   without three quotes?" requires the catalogue price **and** the policy
>   bands; answering "what warranty do I get on a turbocharger cartridge?"
>   requires the catalogue's supplier column **and** the supplier terms.
> - Make the arithmetic non-trivial: unit prices with cents, minimum order
>   quantities, and percentage discounts, so that order totals genuinely require
>   a calculator rather than mental arithmetic.
> - Keep every number internally consistent across the three files.

This corpus is chosen to be maximally unlike the HR one: tabular rather than
prose, arithmetic-heavy, date-sensitive, and spread across three interlocking
documents. It should therefore pull a **different tool selection** out of the
factory - csv_lookup, calculator and current_date, where the HR brief should
pull none. That contrast is the test of core requirement 3.
