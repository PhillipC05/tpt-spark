// Prevents additional console window on Windows in release (GUI mode only).
// Headless mode keeps the console open so the named-pipe server is accessible.
#![cfg_attr(
    all(not(debug_assertions), not(feature = "headless-console")),
    windows_subsystem = "windows"
)]

fn main() {
    let headless = std::env::args().any(|a| a == "--headless")
        || std::env::var("TPT_SPARK_HEADLESS").as_deref() == Ok("1");

    if headless {
        tpt_spark_lib::run_headless_blocking();
    } else {
        tpt_spark_lib::run();
    }
}
