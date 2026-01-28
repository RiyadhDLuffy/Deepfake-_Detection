// Service worker disabled for immediate update
self.addEventListener('install', (e) => {
    self.skipWaiting();
});

self.addEventListener('activate', (e) => {
    console.log('🧹 SW: Clearing all old caches...');
    e.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(keys.map((k) => caches.delete(k)));
        }).then(() => {
            console.log('✨ SW: Cache cleared, claiming clients...');
            return self.clients.claim();
        })
    );
});

self.addEventListener('fetch', (e) => {
    // Explicitly bypass cache
    return;
});
