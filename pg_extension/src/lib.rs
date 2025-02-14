use pgrx::prelude::*;

::pgrx::pg_module_magic!();

#[pg_extern]
fn hello_pg_extension() -> &'static str {
    "Hello, pg_extension"
}

